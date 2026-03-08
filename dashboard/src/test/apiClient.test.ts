import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";

describe("api.getStats", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("maps cue and projection metrics from the stats payload", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          stats: {
            entities: 42,
            relationships: 100,
            episodes: 15,
            entity_type_distribution: { Person: 20, Project: 5 },
            cue_metrics: {
              cue_count: 12,
              episodes_without_cues: 3,
              cue_coverage: 0.8,
              cue_hit_count: 9,
              cue_hit_episode_count: 4,
              cue_hit_episode_rate: 0.3333,
              cue_surfaced_count: 6,
              cue_selected_count: 3,
              cue_used_count: 2,
              cue_near_miss_count: 1,
              avg_policy_score: 0.42,
              avg_projection_attempts: 1.5,
              projected_cue_count: 5,
              cue_to_projection_conversion_rate: 0.4167,
            },
            projection_metrics: {
              state_counts: {
                queued: 1,
                cued: 2,
                cue_only: 3,
                scheduled: 2,
                projecting: 1,
                projected: 5,
                failed: 1,
                dead_letter: 0,
              },
              attempted_episode_count: 6,
              total_attempts: 9,
              failure_count: 1,
              dead_letter_count: 0,
              failure_rate: 1 / 6,
              avg_processing_duration_ms: 180,
              avg_time_to_projection_ms: 3200,
              yield: {
                linked_entity_count: 14,
                relationship_count: 8,
                avg_linked_entities_per_projected_episode: 2.8,
                avg_relationships_per_projected_episode: 1.6,
              },
            },
          },
          topActivated: [],
          topConnected: [],
          growthTimeline: [],
        }),
      }),
    );

    const stats = await api.getStats();

    expect(stats.totalEntities).toBe(42);
    expect(stats.cueMetrics?.cueCoverage).toBe(0.8);
    expect(stats.cueMetrics?.projectedCueCount).toBe(5);
    expect(stats.projectionMetrics?.stateCounts.cueOnly).toBe(3);
    expect(stats.projectionMetrics?.yield.avgRelationshipsPerProjectedEpisode).toBe(1.6);
  });
});
