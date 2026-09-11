/** Streamlit transport; calculations stay in the upstream browser engine. */
export interface ClimateArgs { station_key: string; epw_text: string; station_label: string; pressure_kpa: number; response?: { id: string; result?: unknown; error?: string }; }
let args: ClimateArgs | null = null;
const listeners = new Set<(value: ClimateArgs) => void>();
const pending = new Map<string, { resolve: (value: any) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> }>();
export const isEmbedded = (): boolean => typeof window !== 'undefined' && window.parent !== window;
const post = (type: string, extra: object = {}): void => { if (isEmbedded()) window.parent.postMessage({ isStreamlitMessage: true, type, ...extra }, '*'); };
export const getClimateArgs = (): ClimateArgs | null => args;
export function onClimateArgs(callback: (value: ClimateArgs) => void): () => void {
  listeners.add(callback); if (args) callback(args); return () => { listeners.delete(callback); };
}
let queue: Promise<unknown> = Promise.resolve();
export function callClimate(action: string, payload: unknown): Promise<any> {
  const request = queue.then(() => sendClimate(action, payload));
  queue = request.catch(() => undefined);
  return request;
}
function sendClimate(action: string, payload: unknown): Promise<any> {
  if (!isEmbedded()) return Promise.reject(new Error('Open this tool inside Climate Analysis Pro.'));
  return new Promise((resolve, reject) => {
    const id = crypto.randomUUID();
    const timer = setTimeout(() => { pending.delete(id); reject(new Error('The app did not respond. Please retry.')); }, 120000);
    pending.set(id, { resolve, reject, timer });
    post('streamlit:setComponentValue', { value: { id, action, payload }, dataType: 'json' });
  });
}
export function base64Blob(value: string, type: string): Blob {
  return new Blob([Uint8Array.from(atob(value), c => c.charCodeAt(0))], { type });
}
if (typeof window !== 'undefined' && isEmbedded()) {
  window.addEventListener('message', event => {
    if (event.source !== window.parent || event.data?.type !== 'streamlit:render') return;
    args = event.data.args as ClimateArgs;
    const response = args?.response;
    if (response && pending.has(response.id)) {
      const task = pending.get(response.id)!; clearTimeout(task.timer); pending.delete(response.id);
      if (response.error) task.reject(new Error(response.error)); else task.resolve(response.result);
    }
    for (const callback of listeners) callback(args);
    post('streamlit:setFrameHeight', { height: 1000 });
  });
  post('streamlit:componentReady', { apiVersion: 1 });
  post('streamlit:setFrameHeight', { height: 1000 });
}
