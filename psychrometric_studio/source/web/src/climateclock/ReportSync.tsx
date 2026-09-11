import { useEffect, useRef, useState } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { Chart } from '../chart/render.js';
import { defaultDomain } from '../chart/scales.js';
import type { ComfortZone } from '../comfort/polygon.js';
import { callClimate } from './bridge.js';
import { chartToBase64Png } from '../io/image.js';
import { buildReportPayload } from '../io/report.js';
import type { SessionState } from '../io/project.js';
import type { ExportPanelProps } from '../ui/ExportPanel.js';
import type { EpwFile } from '../weather/epw.js';

type Props = Pick<ExportPanelProps, 'cases' | 'atmosphere' | 'units'> & {
  stationKey: string | null; session: SessionState; weatherFile: EpwFile | null; width: number; height: number;
  zones: readonly ComfortZone[];
};
export function ReportSync(props: Props): React.JSX.Element | null {
  const [status, setStatus] = useState('Preparing the studio...');
  const [retry, setRetry] = useState(0);
  const latest = useRef(props); latest.current = props;
  // Export metadata contains a fresh timestamp; using it here retriggers forever.
  const signature = JSON.stringify(props.session);
  useEffect(() => {
    if (!props.stationKey) return;
    let cancelled = false;
    setStatus('Updating the full climate report...');
    const timer = setTimeout(() => { void (async () => {
      try {
        const p = latest.current;
        const cases = [];
        for (const [index, entry] of p.cases.entries()) {
          // Reports use a readable landscape view and include the whole design,
          // even when the on-screen chart is narrow or panned away from a point.
          const domain = { ...defaultDomain(p.units) };
          const states = entry.solved.stages.flatMap(stage => stage.result ? [stage.result.state] : []);
          for (const state of [...states, ...entry.weather.hours]) {
            if (Number.isFinite(state.tdb)) {
              domain.tdbMin = Math.min(domain.tdbMin, state.tdb - 2);
              domain.tdbMax = Math.max(domain.tdbMax, state.tdb + 2);
            }
            if (Number.isFinite(state.w)) domain.wMax = Math.max(domain.wMax, state.w * 1.1);
          }
          const settings = p.session.systems[index]!;
          const host = document.createElement('div');
          host.innerHTML = renderToStaticMarkup(<Chart domain={domain} pressure={p.atmosphere.pressure} units={p.units} width={1200} height={760}
            visibility={settings.visibility} showProtractor={settings.showProtractor} hover={null} solved={entry.solved} comfortZones={p.zones} designDays={p.weatherFile?.design?.days ?? []} />);
          const svg = host.querySelector('svg');
          if (!svg) throw new Error('Could not render the report chart.');
          const chartPng = await chartToBase64Png({ svg, domain, weather: entry.weather, caption: entry.label }, 1.5);
          cases.push({ ...buildReportPayload({ solved: entry.solved, units: p.units, atmosphere: p.atmosphere, meta: { ...p.session.meta, name: `${p.session.meta.name ?? 'Psychrometric design'} - ${entry.label}` }, chartPng }), weatherStation: p.weatherFile?.location.city ?? '', weatherHours: entry.weather.hours.length, weatherMode: entry.weather.mode, caseLabel: entry.label });
        }
        if (cancelled) return;
        await callClimate('sync_report', { station_key: p.stationKey, cases });
        if (!cancelled) setStatus(`${cases.length} operating cases are included in your full climate report.`);
      } catch (error) { if (!cancelled) setStatus(`Report update failed: ${error instanceof Error ? error.message : error}`); }
    })(); }, 1400);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [signature, props.stationKey, props.weatherFile, props.width, props.height, retry]);
  if (!props.stationKey) return null;
  return <div className="climateclock-sync" role="status"><span>{status}</span><button type="button" onClick={() => setRetry(v => v + 1)}>Update report</button></div>;
}
