import type { EntitySpan } from '../domain/types';

/**
 * Extracts a substring from a raw narrative using Python-style code point indexing.
 * 
 * JavaScript strings are UTF-16 encoded, so characters outside the Basic Multilingual Plane 
 * (like emojis or supplementary characters) count as 2 units (surrogate pairs).
 * Python strings are indexed by true Unicode code points.
 * 
 * To ensure safe evidence highlighting, we must segment strings by code points.
 * 
 * @param text The raw narrative text
 * @param startChar The start code point index (inclusive)
 * @param endChar The end code point index (exclusive)
 * @returns The extracted text segment
 */
export function extractCodePointSpan(text: string, startChar: number, endChar: number): string {
  // Array.from splits a string into true Unicode code points
  const codePoints = Array.from(text);

  if (startChar < 0 || endChar > codePoints.length || startChar >= endChar) {
    console.error(`Invalid span bounds: [${startChar}, ${endChar}] for text length ${codePoints.length}`);
    return '';
  }

  return codePoints.slice(startChar, endChar).join('');
}

export interface Segment {
  text: string;
  isEvidence: boolean;
  spanContext?: EntitySpan;
}

/**
 * Segments a full text into highlighted and non-highlighted pieces based on a list of spans.
 * This is safe against HTML injection because it returns pure data structures, not HTML strings.
 * 
 * @param text The raw text
 * @param spans The array of entity spans
 * @returns Array of text segments
 */
export function segmentTextBySpans(text: string, spans: EntitySpan[]): Segment[] {
  if (!spans || spans.length === 0) {
    return [{ text, isEvidence: false }];
  }

  // Sort spans by start index
  const sortedSpans = [...spans].sort((a, b) => a.start_char - b.start_char);

  const segments: Segment[] = [];
  let currentIndex = 0;

  for (const span of sortedSpans) {
    // Detect and skip overlapping spans or invalid bounds
    if (span.start_char < currentIndex) {
      console.warn(`Overlapping or out-of-order span detected: ${span.canonical_form}`);
      continue;
    }

    if (span.start_char > text.length || span.end_char > text.length) {
      console.error(`Span out of bounds: [${span.start_char}, ${span.end_char}]`);
      continue;
    }

    // Add unhighlighted text before the span
    if (span.start_char > currentIndex) {
      segments.push({
        text: text.slice(currentIndex, span.start_char),
        isEvidence: false
      });
    }

    // Add the highlighted span
    segments.push({
      text: text.slice(span.start_char, span.end_char),
      isEvidence: true,
      spanContext: span
    });

    currentIndex = span.end_char;
  }

  // Add any remaining unhighlighted text
  if (currentIndex < text.length) {
    segments.push({
      text: text.slice(currentIndex),
      isEvidence: false
    });
  }

  return segments;
}
