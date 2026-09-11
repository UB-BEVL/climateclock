import { describe, expect, it } from 'vitest';
import { starterSystems } from '../src/climateclock/starterSystems.js';
import { standardAtmosphere } from '../src/psych/atmosphere.js';
import { solveSystem } from '../src/processes/chain.js';

describe('ClimateClock SI starter cases', () => {
  it('preserves the physical outdoor condition when starting in SI', () => {
    const systems = starterSystems('SI');
    for (const system of systems) {
      const result = solveSystem({airstreams: [{id: 'supply', name: 'Supply air', stages: system.stages}]}, standardAtmosphere('SI').pressure, 'SI');
      for (const stage of result.airstreams[0]!.stages) expect(stage.error).toBeUndefined();
      const temperature = result.airstreams[0]!.stages[0]!.result!.state.tdb;
      expect(temperature).toBeCloseTo(system.id === 'cooling' ? 35 : -15, 6);
    }
  });
});
