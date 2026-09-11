import { blankSystem, type SessionSystem } from '../io/project.js';
import { convertStages } from '../ui/convertProject.js';
import { STARTER_COOLING, STARTER_HEATING } from '../ui/starters.js';
import type { Stage } from '../types/project.js';
import type { UnitSystem } from '../psych/units.js';

export function starterSystems(units: UnitSystem): SessionSystem[] {
  const convert = (stages: readonly Stage[]): Stage[] => units === 'IP' ? [...stages] : convertStages([...stages], 'IP', units);
  return [blankSystem('cooling', units, convert(STARTER_COOLING)), blankSystem('heating', units, convert(STARTER_HEATING))];
}
