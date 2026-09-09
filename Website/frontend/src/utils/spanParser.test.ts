import { describe, it, expect } from 'vitest';
import { extractCodePointSpan, segmentTextBySpans } from './spanParser';
import type { EntitySpan } from '../domain/types';

describe('extractCodePointSpan', () => {
  it('handles standard ASCII strings', () => {
    const text = 'Worker entered confined space';
    // "confined space" is from 15 to 29
    expect(extractCodePointSpan(text, 15, 29)).toBe('confined space');
  });

  it('handles strings with emojis (UTF-16 surrogate pairs)', () => {
    // 🚧 is 2 UTF-16 code units, but 1 code point.
    const text = 'Worker entered confined space 🚧 without testing.';

    // In Python, 🚧 is at index 30, ' ' is at 31, 'w' is at 32
    // Length up to space: "Worker entered confined space " = 30 chars.
    // 🚧 = index 30. " " = index 31. "without testing." starts at 32.

    expect(extractCodePointSpan(text, 30, 31)).toBe('🚧');
    expect(extractCodePointSpan(text, 32, 39)).toBe('without');
  });

  it('handles out of bound indices gracefully', () => {
    const text = 'Short';
    expect(extractCodePointSpan(text, -1, 10)).toBe('');
    expect(extractCodePointSpan(text, 5, 2)).toBe('');
  });
});

describe('segmentTextBySpans', () => {
  it('segments text correctly with valid non-overlapping spans', () => {
    const text = 'Worker bypassed LOTO protocol today.';
    const spans: EntitySpan[] = [
      {
        text: 'bypassed',
        canonical_form: 'bypass',
        start_char: 7,
        end_char: 15,
        entity_type: 'FAILED_BARRIER'
      },
      {
        text: 'LOTO protocol',
        canonical_form: 'energy_isolation',
        start_char: 16,
        end_char: 29,
        entity_type: 'BARRIER'
      }
    ];

    const segments = segmentTextBySpans(text, spans);

    expect(segments.length).toBe(5);
    expect(segments[0]).toEqual({ text: 'Worker ', isEvidence: false });
    expect(segments[1].text).toBe('bypassed');
    expect(segments[1].isEvidence).toBe(true);
    expect(segments[2]).toEqual({ text: ' ', isEvidence: false });
    expect(segments[3].text).toBe('LOTO protocol');
    expect(segments[3].isEvidence).toBe(true);
    expect(segments[4]).toEqual({ text: ' today.', isEvidence: false });
  });

  it('handles overlapping spans by ignoring the overlapping portion safely', () => {
    const text = 'Worker bypassed LOTO protocol today.';
    const spans: EntitySpan[] = [
      {
        text: 'bypassed LOTO',
        canonical_form: 'bypass_loto',
        start_char: 7,
        end_char: 20,
        entity_type: 'FAILED_BARRIER'
      },
      {
        text: 'LOTO', // Overlaps with previous
        canonical_form: 'loto',
        start_char: 16,
        end_char: 20,
        entity_type: 'BARRIER'
      }
    ];

    const segments = segmentTextBySpans(text, spans);

    // It should skip the second span because its start_char < currentIndex
    expect(segments.length).toBe(3);
    expect(segments[1].text).toBe('bypassed LOTO');
    expect(segments[1].isEvidence).toBe(true);
  });
});
