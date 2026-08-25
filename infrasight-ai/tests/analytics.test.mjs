import test from 'node:test';
import assert from 'node:assert/strict';
import { createSceneSummary, formatPercent, summarizeDetections } from '../js/analytics.js';

test('summarizeDetections groups transportation classes', () => {
  const result = summarizeDetections([
    { class: 'car', score: 0.92 },
    { class: 'person', score: 0.81 },
    { class: 'truck', score: 0.77 },
    { class: 'airplane', score: 0.88 }
  ]);
  assert.equal(result.objects, 4);
  assert.equal(result.vehicles, 2);
  assert.equal(result.people, 1);
  assert.equal(result.aircraft, 1);
  assert.equal(result.counts.car, 1);
});

test('empty scene summary is explicit', () => {
  const summary = summarizeDetections([]);
  assert.equal(createSceneSummary(summary), 'No objects met the selected confidence threshold.');
});

test('formatPercent rounds confidence values', () => {
  assert.equal(formatPercent(0.876), '88%');
});
