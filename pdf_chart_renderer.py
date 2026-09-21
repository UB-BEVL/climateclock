"""Render PDF charts in an isolated, reusable process with the project's packages.

Only standard-library imports run in the application process. This lets a local
app accidentally started with another Python still use its installed renderer.
"""
from __future__ import annotations

import base64
from contextvars import ContextVar
from functools import wraps
import json
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
import tempfile
from threading import Thread
import uuid


_active_renderer = ContextVar('climateclock_pdf_renderer', default=None)


class ChartRenderError(RuntimeError):
    pass


class ChartRenderer:
    def __init__(self, root=None, *, timeout=120):
        self.root = Path(root) if root is not None else Path(__file__).resolve().parent
        self.timeout = timeout
        self.process = None
        self.reader = None
        self.responses = Queue()
        self.directory = tempfile.TemporaryDirectory(prefix='climateclock-pdf-')
        self.error_log = tempfile.TemporaryFile(mode='w+b')

    @property
    def python(self):
        private = self.root / '.venv-climateclock' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        return str(private) if private.is_file() else sys.executable

    def __enter__(self):
        try:
            self.process = subprocess.Popen(
                [self.python, '-u', str(Path(__file__).resolve()), '--worker'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.error_log,
                text=True, encoding='utf-8',
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0,
            )
            self.reader = Thread(target=self._read_responses, daemon=True)
            self.reader.start()
            # Exercise the actual image engine before building any report pages.
            self._request({'figure': '{"data":[{"type":"scatter","x":[0,1],"y":[0,1]}]}',
                           'width': 200, 'height': 150, 'scale': 1})
            return self
        except Exception as exc:
            self.close()
            raise ChartRenderError(
                'The PDF chart renderer could not start. From the climateclock folder, '
                'run `py run_app.py --setup-only`, then start with `py run_app.py`. '
                f'Renderer error: {exc}'
            ) from exc

    def _read_responses(self):
        try:
            for line in self.process.stdout:
                self.responses.put(json.loads(line))
        except Exception as exc:
            self.responses.put({'error': str(exc)})
        finally:
            self.responses.put({'error': 'The chart-rendering process stopped unexpectedly.'})

    def _request(self, payload):
        try:
            self.process.stdin.write(json.dumps(payload, ensure_ascii=True) + '\n')
            self.process.stdin.flush()
            response = self.responses.get(timeout=self.timeout)
        except Empty as exc:
            # Never let a late image response be mistaken for the next chart.
            self.close()
            raise ChartRenderError(f'Chart rendering exceeded {self.timeout} seconds.') from exc
        except (OSError, ValueError) as exc:
            raise ChartRenderError(f'Could not communicate with the chart renderer: {exc}') from exc
        if response.get('error'):
            raise ChartRenderError(response['error'])
        data = base64.b64decode(response.get('png', ''), validate=True)
        if not data.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ChartRenderError('The renderer returned an invalid chart image.')
        return data

    def render(self, figure, width, height, scale):
        return self._request({'figure': figure.to_json(), 'width': width, 'height': height, 'scale': scale})

    def close(self):
        process = self.process
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if self.reader is not None:
                self.reader.join(timeout=2)
            if process.stdout is not None:
                process.stdout.close()
        self.error_log.close()
        self.directory.cleanup()

    def __exit__(self, *args):
        self.close()


def report_chart_session(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with ChartRenderer() as renderer:
            token = _active_renderer.set(renderer)
            try:
                return function(*args, **kwargs)
            finally:
                _active_renderer.reset(token)
    return wrapped


def render_chart_png(figure, width, height, scale):
    renderer = _active_renderer.get()
    if renderer is None:
        # Standalone chart exports still work outside a full report.
        with ChartRenderer() as standalone:
            return standalone.render(figure, width, height, scale)
    return renderer.render(figure, width, height, scale)


def save_chart_png(data):
    renderer = _active_renderer.get()
    if renderer is not None:
        path = Path(renderer.directory.name) / f'{uuid.uuid4().hex}.png'
        path.write_bytes(data)
        return str(path)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.kaleido.png') as target:
        target.write(data)
        return target.name


def _decode_plotly_arrays(value):
    """Expand Plotly 6 binary arrays for the pinned Plotly 5 renderer."""
    if isinstance(value, list):
        return [_decode_plotly_arrays(item) for item in value]
    if isinstance(value, dict):
        if 'bdata' in value and 'dtype' in value:
            import numpy as np
            dtype = np.dtype(value['dtype'])
            if dtype.kind not in 'biuf':
                raise ValueError('Unsupported Plotly array type.')
            array = np.frombuffer(base64.b64decode(value['bdata'], validate=True), dtype=dtype)
            if value.get('shape'):
                array = array.reshape(tuple(int(part) for part in value['shape'].split(',')))
            return array.tolist()
        return {key: _decode_plotly_arrays(item) for key, item in value.items()}
    return value


def _restore_numeric_marker_gaps(figure):
    """Restore NaN colours encoded as JSON null by Plotly 5.

    Plotly accepts numeric NaN as missing data, but rejects None in a colour
    array when validating a figure reconstructed from JSON. Preserve gaps and
    finite values without replacing unknown measurements with made-up numbers.
    Leave categorical/invalid colour arrays to normal Plotly validation.
    """
    for trace in figure.get('data', []):
        marker = trace.get('marker', {})
        for container in (marker, marker.get('line', {})):
            colors = container.get('color')
            if (isinstance(colors, list) and any(value is None for value in colors)
                    and all(value is None or isinstance(value, (int, float)) for value in colors)):
                container['color'] = [float('nan') if value is None else value for value in colors]
    return figure


def _worker():
    import plotly.io as pio
    scope = getattr(pio.kaleido, 'scope', None)
    if scope is not None:
        scope.mathjax = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                figure = _restore_numeric_marker_gaps(_decode_plotly_arrays(json.loads(request['figure'])))
                # Invalid properties fail visibly instead of silently changing a chart.
                png = pio.to_image(figure, format='png', width=request['width'],
                                   height=request['height'], scale=request['scale'])
                response = {'png': base64.b64encode(png).decode('ascii')}
            except Exception as exc:
                response = {'error': str(exc)[:1200]}
            print(json.dumps(response), flush=True)
    finally:
        # Kaleido 0.2 drops its process without closing both output pipes.
        process = getattr(scope, '_proc', None)
        if scope is not None and hasattr(scope, '_shutdown_kaleido'):
            scope._shutdown_kaleido()
        thread = getattr(scope, '_std_error_thread', None)
        if thread is not None:
            thread.join(timeout=2)
        if process is not None and process.poll() is not None:
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()


if __name__ == '__main__' and sys.argv[1:] == ['--worker']:
    _worker()
