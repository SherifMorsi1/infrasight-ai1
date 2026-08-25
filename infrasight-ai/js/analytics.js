export const VEHICLE_CLASSES = new Set([
  'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck'
]);

export const AVIATION_CLASSES = new Set(['airplane']);

export function summarizeDetections(predictions = []) {
  const counts = new Map();
  let confidenceTotal = 0;
  let vehicles = 0;
  let people = 0;
  let aircraft = 0;

  for (const prediction of predictions) {
    const className = String(prediction.class || 'unknown').toLowerCase();
    const score = Number(prediction.score || 0);
    counts.set(className, (counts.get(className) || 0) + 1);
    confidenceTotal += score;
    if (VEHICLE_CLASSES.has(className)) vehicles += 1;
    if (className === 'person') people += 1;
    if (AVIATION_CLASSES.has(className)) aircraft += 1;
  }

  const sortedClasses = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  const topClass = sortedClasses.length ? sortedClasses[0][0] : null;

  return {
    objects: predictions.length,
    vehicles,
    people,
    aircraft,
    averageConfidence: predictions.length ? confidenceTotal / predictions.length : 0,
    topClass,
    counts: Object.fromEntries(sortedClasses)
  };
}

export function createSceneSummary(summary) {
  if (!summary || summary.objects === 0) {
    return 'No objects met the selected confidence threshold.';
  }

  const parts = [];
  if (summary.aircraft > 0) {
    parts.push(`${summary.aircraft} aircraft ${summary.aircraft === 1 ? 'object' : 'objects'} detected`);
  }
  if (summary.vehicles > 0) {
    parts.push(`${summary.vehicles} mobility ${summary.vehicles === 1 ? 'object' : 'objects'}`);
  }
  if (summary.people > 0) {
    parts.push(`${summary.people} ${summary.people === 1 ? 'person' : 'people'}`);
  }

  if (!parts.length) {
    return `${summary.objects} ${summary.objects === 1 ? 'object was' : 'objects were'} detected, led by ${summary.topClass}.`;
  }

  return `${capitalize(parts.join(', '))}. ${summary.objects} total ${summary.objects === 1 ? 'detection' : 'detections'} passed the threshold.`;
}

export function formatPercent(value) {
  const safe = Number.isFinite(Number(value)) ? Number(value) : 0;
  return `${Math.round(safe * 100)}%`;
}

export function capitalize(text) {
  const value = String(text || '');
  return value ? value.charAt(0).toUpperCase() + value.slice(1) : value;
}
