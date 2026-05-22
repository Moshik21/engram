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
            adjudication_metrics: {
              evidence_status_counts: { pending: 1, deferred: 2, approved: 0 },
              request_status_counts: { pending: 1, deferred: 0, error: 1 },
              open_evidence_count: 3,
              pending_evidence_count: 1,
              deferred_evidence_count: 2,
              approved_evidence_count: 0,
              open_request_count: 2,
              pending_request_count: 1,
              deferred_request_count: 0,
              error_request_count: 1,
              open_work_count: 5,
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
    expect(stats.adjudicationMetrics?.openWorkCount).toBe(5);
    expect(stats.adjudicationMetrics?.evidenceStatusCounts.deferred).toBe(2);
  });
});

describe("api.getStorage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("loads the shared storage report", async () => {
    const payload = {
      mode: "helix",
      configuredMode: "helix",
      backend: "helix_native",
      groupId: "default",
      startedAt: "2026-05-19T12:00:00Z",
      uptimeSeconds: 10,
      counts: { episodes: 3, entities: 4, relationships: 5, cues: 2 },
      startupCounts: { episodes: 1, entities: 2, relationships: 3, cues: 1 },
      growthSinceStartup: {
        bytes: 2048,
        episodes: 2,
        entities: 2,
        relationships: 2,
        cues: 1,
      },
      disk: {
        totalBytes: 4096,
        humanSize: "4.0 KB",
        startupBytes: 2048,
        startupHumanSize: "2.0 KB",
      },
      paths: [
        {
          label: "Helix native data",
          path: "/tmp/engram-native",
          exists: true,
          kind: "directory",
          bytes: 4096,
          humanSize: "4.0 KB",
        },
      ],
    };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => payload,
    });
    vi.stubGlobal("fetch", fetchMock);

    const storage = await api.getStorage();

    expect(fetchMock).toHaveBeenCalledWith("/api/storage", expect.any(Object));
    expect(storage.backend).toBe("helix_native");
    expect(storage.paths[0].path).toBe("/tmp/engram-native");
  });
});

describe("api.getRuntimeState", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("loads runtime adoption guidance from the shared runtime endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        projectName: "engram",
        runtime: { mode: "helix" },
        activation: { recallProfile: "wave1" },
        features: { artifactBootstrapEnabled: true },
        artifactBootstrap: {
          enabled: true,
          projectPath: "/tmp/engram",
          artifactCount: 0,
          freshArtifactCount: 0,
          staleArtifactCount: 0,
          lastObservedAt: null,
          staleAfterSeconds: 86400,
        },
        agentAdoption: {
          status: "fresh_runtime",
          doNotTreatEmptyAsFailure: true,
          requiredNextTools: ["claim_authority", "bootstrap_project", "get_context"],
          beforeAnswer: {
            required: true,
            tools: ["claim_authority", "bootstrap_project", "get_context"],
            reason: "Follow these Engram tools before the next substantive answer.",
          },
          emptyRuntimePolicy: "Treat an empty or fresh Engram runtime as onboarding state.",
          fileMemoryPolicy: "Project-local memory files do not replace Engram.",
          claimAuthority: {
            tool: "claim_authority",
            args: {
              project_path: "/tmp/engram",
              file_memory_present: "<true if local/file memory is visible>",
            },
            reason: "Ask Engram for the source-of-truth contract.",
          },
          bootstrap: {
            tool: "bootstrap_project",
            required: true,
            args: { project_path: "/tmp/engram" },
            reason: "Fresh runtime is onboarding state.",
          },
          reason: "Connected but empty/fresh Engram runtime.",
        },
        stats: { recallMetrics: {}, epistemicMetrics: {} },
        generatedAt: "2026-05-19T00:00:00Z",
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const runtime = await api.getRuntimeState({ projectPath: "/tmp/engram" });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/knowledge/runtime?project_path=%2Ftmp%2Fengram",
      expect.any(Object),
    );
    expect(runtime.agentAdoption.status).toBe("fresh_runtime");
    expect(runtime.agentAdoption.requiredNextTools).toEqual([
      "claim_authority",
      "bootstrap_project",
      "get_context",
    ]);
    expect(runtime.agentAdoption.beforeAnswer?.required).toBe(true);
    expect(runtime.agentAdoption.emptyRuntimePolicy).toContain("onboarding state");
  });
});

describe("api.getLifecycleSummary", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("loads the backend brain-loop summary contract", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        groupId: "default",
        generatedAt: "2026-05-11T12:00:00Z",
        loop: ["capture", "cue", "project", "recall", "consolidate"],
        totals: { episodes: 0, cues: 0, projected: 0, cycles: 0, entities: 0, relationships: 0 },
        capture: { status: "ready", episodeCount: 0, activeCount: 0, latestEpisode: null },
        cue: {
          status: "ready",
          cueCount: 0,
          episodesWithoutCues: 0,
          coverage: 0,
          hitCount: 0,
          surfacedCount: 0,
          selectedCount: 0,
          usedCount: 0,
          nearMissCount: 0,
          avgPolicyScore: 0,
          projectionConversionRate: 0,
        },
        project: {
          status: "ready",
          projectedCount: 0,
          activeCount: 0,
          failedCount: 0,
          deadLetterCount: 0,
          failureRate: 0,
          stateCounts: {
            queued: 0,
            cued: 0,
            cueOnly: 0,
            scheduled: 0,
            projecting: 0,
            projected: 0,
            merged: 0,
            failed: 0,
            deadLetter: 0,
          },
        },
        recall: {
          status: "ready",
          activeEntityCount: 0,
          topScore: 0,
          triggerCount: 0,
          intentions: {
            activeCount: 0,
            refreshContextCount: 0,
            afterConsolidationCount: 0,
            pinnedResultCount: 0,
            needsRefreshCount: 0,
            latestRefreshedAt: null,
          },
          topActivated: [],
        },
        consolidate: {
          status: "ready",
          isRunning: false,
          schedulerActive: false,
          cycleCount: 0,
          pressure: null,
          latestCycle: null,
        },
        recentEpisodes: [],
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const summary = await api.getLifecycleSummary();

    expect(fetchMock).toHaveBeenCalledWith("/api/lifecycle/summary", expect.any(Object));
    expect(summary.loop).toEqual(["capture", "cue", "project", "recall", "consolidate"]);
    expect(summary.recall.intentions.activeCount).toBe(0);
  });
});

describe("api.getEvaluationReport", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("maps the backend evaluation report contract", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        group_id: "default",
        generated_at: "2026-05-11T12:00:00Z",
        degraded: true,
        degradations: [
          {
            stage: "graph_state",
            status: "degraded",
            skip_reason: "graph_state_timeout",
            timeout_ms: 2000,
          },
        ],
        loop: ["capture", "cue", "project", "recall", "consolidate"],
        totals: { episodes: 5, entities: 3, relationships: 2, active_entities: 1 },
        capture: { status: "ready", episode_count: 5, active_count: 0 },
        cue: {
          status: "ready",
          cue_count: 4,
          episodes_without_cues: 1,
          coverage: 0.8,
          hit_count: 8,
          hit_episode_count: 3,
          hit_episode_rate: 0.75,
          surfaced_count: 6,
          selected_count: 3,
          used_count: 2,
          near_miss_count: 1,
          selected_rate: 0.5,
          used_rate: 0.3333,
          near_miss_rate: 0.1667,
          avg_policy_score: 0.7,
          projection_conversion_rate: 0.5,
        },
        project: {
          status: "attention",
          state_counts: { projected: 2, failed: 1, dead_letter: 0, cue_only: 1 },
          tracked_count: 5,
          projected_count: 2,
          active_count: 1,
          projected_rate: 0.4,
          backlog_rate: 0.2,
          failed_count: 1,
          dead_letter_count: 0,
          attempted_episode_count: 3,
          total_attempts: 4,
          failure_rate: 0.3333,
          avg_processing_duration_ms: 42,
          avg_time_to_projection_ms: 1500,
          yield: {
            linked_entity_count: 6,
            relationship_count: 3,
            avg_linked_entities_per_projected_episode: 3,
            avg_relationships_per_projected_episode: 1.5,
          },
        },
        recall: {
          status: "active",
          total_analyses: 7,
          trigger_count: 4,
          runtime_false_recall_rate: 0.25,
          runtime_surfaced_to_used_ratio: 3,
          graph_lift_rate: 0.1,
          probe_trigger_rate: 0.2,
          latency: {
            analyzer_ms: { avg_ms: 12, p95_ms: 31 },
            probe_ms: { avg: 7, p95: 19 },
          },
          control: {
            used_count: 3,
            dismissed_count: 1,
            surfaced_count: 5,
            selected_count: 2,
            confirmed_count: 1,
            corrected_count: 1,
            graph_override_count: 2,
            adaptive_thresholds_enabled: true,
            thresholds: { linguistic: 0.32, borderline: 0.18, resonance: 0.5 },
          },
          family_contributions: { linguistic: 2 },
          evaluation: {
            status: "measured",
            sample_count: 2,
            need_status: "measured",
            need_labeled_count: 2,
            needed_count: 2,
            missed_count: 1,
            memory_need_precision: 0.5,
            memory_need_recall: 0.5,
            missed_recall_rate: 0.5,
            useful_packet_rate: 0.4,
            stale_packet_rate: 0.2,
            corrected_packet_rate: 0.1,
            stale_packet_count: 1,
            corrected_packet_count: 1,
            false_recall_rate: 0.2,
            surfaced_count: 5,
            used_count: 2,
            surfaced_to_used_ratio: 2.5,
          },
          continuity: {
            status: "measured",
            sample_count: 1,
            session_continuity_lift: 0.3,
            open_loop_recovery_rate: 1,
            temporal_correctness: 0,
          },
        },
        memory_value: {
          status: "measured",
          cost: {
            status: "measured",
            operation_count: 6,
            avg_added_latency_ms: 9.5,
            p95_added_latency_ms: 24,
            avg_budget_ms: 600,
            p95_budget_ms: 1200,
            avg_budget_tokens: 300,
            completed_count: 4,
            skipped_count: 1,
            error_count: 1,
            status_counts: { ok: 4, skipped: 1, error: 1 },
            skip_reason_counts: { skipped_low_signal: 1 },
            timeout_count: 1,
            timeout_rate: 0.1667,
            degraded_count: 0,
            degraded_rate: 0,
            budget_miss_count: 2,
            budget_miss_rate: 0.3333,
            cache_hit_count: 3,
            cache_miss_count: 3,
            cache_hit_rate: 0.5,
            by_mode: {
              cached: {
                status: "measured",
                operation_count: 3,
                avg_added_latency_ms: 3,
                p95_added_latency_ms: 6,
                skipped_count: 0,
                error_count: 0,
                cache_hit_count: 3,
                cache_miss_count: 0,
                cache_hit_rate: 1,
              },
            },
          },
          benefit: {
            status: "measured",
            recall_sample_count: 2,
            session_sample_count: 1,
            memory_need_precision: 0.5,
            memory_need_recall: 0.5,
            useful_packet_rate: 0.4,
            stale_packet_rate: 0.2,
            corrected_packet_rate: 0.1,
            stale_packet_count: 1,
            corrected_packet_count: 1,
            false_recall_rate: 0.2,
            session_continuity_lift: 0.3,
            open_loop_recovery_rate: 1,
            temporal_correctness: 0,
          },
        },
        consolidate: {
          status: "attention",
          cycle_count: 1,
          latest_status: "completed",
          latest_cycle: {
            id: "cyc_1",
            error: "calibration failed",
            phase_issue: "calibrate: no teacher labels",
          },
          phase_status_counts: { success: 2 },
          phase_totals: {
            triage: { runs: 1, items_processed: 4, items_affected: 2, effect_rate: 0.5 },
          },
          adjudication: {
            status: "active",
            phase_count: 1,
            runs: 1,
            items_processed: 3,
            items_affected: 1,
            items_unaffected: 2,
            effect_rate: 0.3333,
            error_count: 0,
            open_evidence_count: 2,
            open_request_count: 1,
            open_work_count: 3,
            pending_evidence_count: 1,
            deferred_evidence_count: 1,
            approved_evidence_count: 0,
            pending_request_count: 1,
            deferred_request_count: 0,
            error_request_count: 0,
            evidence_status_counts: { pending: 1, deferred: 1, approved: 0 },
            request_status_counts: { pending: 1, deferred: 0, error: 0 },
            phase_totals: {
              edge_adjudication: {
                runs: 1,
                items_processed: 3,
                items_affected: 1,
                effect_rate: 0.3333,
              },
            },
          },
          calibration: {
            status: "measured",
            snapshot_count: 1,
            phase_totals: {
              triage: {
                snapshots: 1,
                total_traces: 5,
                labeled_examples: 3,
                oracle_examples: 1,
                abstain_count: 0,
                accuracy: 0.67,
                mean_confidence: 0.8,
                expected_calibration_error: 0.12,
              },
            },
          },
          items_processed: 4,
          items_affected: 2,
          effect_rate: 0.5,
          error_count: 0,
        },
        evaluation_signals: {
          cue_usefulness: {
            status: "measured",
            evidence_count: 6,
            metric: 0.3333,
            gap: null,
          },
          projection_yield: {
            status: "measured",
            evidence_count: 2,
            metric: 3,
            gap: null,
          },
          recall_quality: {
            status: "measured",
            evidence_count: 2,
            metric: 0.5,
            gap: null,
          },
          false_recall: {
            status: "measured",
            evidence_count: 5,
            metric: 0.2,
            gap: null,
          },
          triage_calibration: {
            status: "measured",
            evidence_count: 3,
            metric: 0.12,
            gap: null,
          },
          consolidation_effect: {
            status: "measured",
            evidence_count: 1,
            metric: 0.5,
            gap: null,
          },
        },
        release_evidence: {
          status: "measured",
          components: {
            evaluation_signals: { status: "measured", missing: [], failures: [] },
            human_labels: { status: "measured", missing: [], failures: [] },
            adoption: {
              status: "measured",
              missing: [],
              failures: [],
              blockers: ["authentication_failed"],
              blocker_details: ["system:error: Not logged in - Please run /login"],
              mcp_server_failures: ["engram"],
            },
            adoption_clients: {
              status: "measured",
              missing: [],
              failures: [],
              required_clients: ["Cursor", "Windsurf"],
              observed_clients: ["Cursor", "Windsurf"],
              blockers: ["authentication_failed"],
              mcp_server_failures: ["engram"],
            },
          },
          missing: [],
          failures: [],
        },
        human_label_evidence: {
          status: "measured",
          artifact_path: "human-labels.json",
          artifact_sha256: "human123",
          kind: "engram_human_label_evidence",
          source: "staging_harness",
          client: "Cursor",
          captured_at: "2026-05-18T23:00:00Z",
          session_id: "cursor-thread-1",
          labeler: "operator",
          human_labeled: true,
          recall_sample_count: 12,
          session_sample_count: 4,
          min_recall_samples: 10,
          min_session_samples: 3,
          sample_sources: ["staging_harness"],
          failures: [],
        },
        adoption_evidence: {
          status: "measured",
          artifact_path: "cursor-adoption-report.json",
          artifact_sha256: "adoption123",
          adoption_status: "passed",
          authority_path: "authority.json",
          calls_path: "calls.jsonl",
          call_count: 4,
          client: "Cursor",
          required_client: "Cursor",
          gate_required_client: "Cursor",
          captured_at: "2026-05-18T23:00:00Z",
          session_id: "cursor-thread-1",
          session_filter: "cursor-thread-1",
          source: "live_harness",
          required_live_evidence: true,
          blockers: ["authentication_failed"],
          blocker_details: ["system:error: Not logged in - Please run /login"],
          mcp_server_failures: ["engram"],
          required_tools: {
            expected: ["get_context", "recall", "observe"],
            observed: ["get_context", "recall", "observe"],
            missing: [],
            in_order: true,
          },
          capture: {
            destination: "engram",
            expected_tool: "observe",
            observed_tools: ["observe"],
            missing: false,
          },
          file_memory: { present: false, substituted_for_engram: false },
          failures: [],
        },
        additional_adoption_evidence: [
          {
            status: "measured",
            client: "Windsurf",
            call_count: 3,
            blockers: ["mcp_server_failed"],
            mcp_server_failures: ["engram"],
            required_tools: { expected: ["get_context"], observed: ["get_context"] },
            capture: { observed_tools: ["observe"] },
            file_memory: { substituted_for_engram: false },
            failures: [],
          },
        ],
        adoption_client_evidence: {
          status: "measured",
          required_clients: ["Cursor", "Windsurf"],
          observed_clients: ["Cursor", "Windsurf"],
          report_count: 2,
          reports: [
            {
              client: "Cursor",
              required_client: "Cursor",
              status: "measured",
              artifact_path: "cursor-adoption-report.json",
              artifact_sha256: "cursor123",
              captured_at: "2026-05-18T23:00:00Z",
              session_id: "cursor-thread-1",
              failures: [],
            },
            {
              client: "Windsurf",
              required_client: "Windsurf",
              status: "measured",
              artifact_path: "windsurf-adoption-report.json",
              artifact_sha256: "windsurf123",
              captured_at: "2026-05-18T23:01:00Z",
              session_id: "windsurf-thread-1",
              blockers: ["mcp_server_failed"],
              blocker_details: ["mcp server engram failed"],
              mcp_server_failures: ["engram"],
              failures: [],
            },
          ],
          blockers: ["mcp_server_failed"],
          mcp_server_failures: ["engram"],
          failures: [],
        },
        coverage_gaps: [],
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const report = await api.getEvaluationReport();

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/evaluation/brain-loop/report",
      expect.any(Object),
    );
    expect(report.groupId).toBe("default");
    expect(report.degraded).toBe(true);
    expect(report.degradations).toEqual([
      {
        surface: null,
        stage: "graph_state",
        status: "degraded",
        skipReason: "graph_state_timeout",
        timeoutMs: 2000,
      },
    ]);
    expect(report.cue.usedRate).toBe(0.3333);
    expect(report.project.stateCounts.cueOnly).toBe(1);
    expect(report.project.trackedCount).toBe(5);
    expect(report.project.projectedRate).toBe(0.4);
    expect(report.project.backlogRate).toBe(0.2);
    expect(report.project.avgProcessingDurationMs).toBe(42);
    expect(report.project.avgTimeToProjectionMs).toBe(1500);
    expect(report.recall.evaluation.memoryNeedPrecision).toBe(0.5);
    expect(report.recall.evaluation.memoryNeedRecall).toBe(0.5);
    expect(report.recall.evaluation.missedRecallRate).toBe(0.5);
    expect(report.recall.evaluation.stalePacketRate).toBe(0.2);
    expect(report.recall.evaluation.correctedPacketRate).toBe(0.1);
    expect(report.recall.latency.analyzerMs.p95Ms).toBe(31);
    expect(report.recall.latency.probeMs.avgMs).toBe(7);
    expect(report.recall.control.graphOverrideCount).toBe(2);
    expect(report.recall.control.thresholds.resonance).toBe(0.5);
    expect(report.recall.continuity.sessionContinuityLift).toBe(0.3);
    expect(report.memoryValue.status).toBe("measured");
    expect(report.memoryValue.cost.operationCount).toBe(6);
    expect(report.memoryValue.cost.p95AddedLatencyMs).toBe(24);
    expect(report.memoryValue.cost.p95BudgetMs).toBe(1200);
    expect(report.memoryValue.cost.skippedCount).toBe(1);
    expect(report.memoryValue.cost.errorCount).toBe(1);
    expect(report.memoryValue.cost.skipReasonCounts.skipped_low_signal).toBe(1);
    expect(report.memoryValue.cost.budgetMissRate).toBe(0.3333);
    expect(report.memoryValue.cost.byMode.cached.cacheHitRate).toBe(1);
    expect(report.memoryValue.benefit.recallSampleCount).toBe(2);
    expect(report.memoryValue.benefit.stalePacketRate).toBe(0.2);
    expect(report.memoryValue.benefit.correctedPacketRate).toBe(0.1);
    expect(report.memoryValue.benefit.sessionContinuityLift).toBe(0.3);
    expect(report.consolidate.phaseTotals.triage.itemsAffected).toBe(2);
    expect(report.consolidate.phaseTotals.triage.effectRate).toBe(0.5);
    expect(report.consolidate.adjudication.itemsUnaffected).toBe(2);
    expect(report.consolidate.adjudication.effectRate).toBe(0.3333);
    expect(report.consolidate.adjudication.openWorkCount).toBe(3);
    expect(report.consolidate.adjudication.evidenceStatusCounts?.deferred).toBe(1);
    expect(report.consolidate.status).toBe("attention");
    expect(report.consolidate.effectRate).toBe(0.5);
    expect(report.consolidate.calibration.phaseTotals.triage.expectedCalibrationError).toBe(0.12);
    expect(report.evaluationSignals.falseRecall).toEqual({
      status: "measured",
      evidenceCount: 5,
      metric: 0.2,
      gap: null,
    });
    expect(report.evaluationSignals.triageCalibration.metric).toBe(0.12);
    expect(report.releaseEvidence?.status).toBe("measured");
    expect(report.releaseEvidence?.components.adoptionClients.requiredClients).toEqual([
      "Cursor",
      "Windsurf",
    ]);
    expect(report.releaseEvidence?.components.adoption.blockers).toEqual([
      "authentication_failed",
    ]);
    expect(report.releaseEvidence?.components.adoption.mcpServerFailures).toEqual([
      "engram",
    ]);
    expect(report.humanLabelEvidence?.client).toBe("Cursor");
    expect(report.humanLabelEvidence?.recallSampleCount).toBe(12);
    expect(report.humanLabelEvidence?.minSessionSamples).toBe(3);
    expect(report.adoptionEvidence?.client).toBe("Cursor");
    expect(report.adoptionEvidence?.requiredTools.observed).toEqual([
      "get_context",
      "recall",
      "observe",
    ]);
    expect(report.adoptionEvidence?.capture.observedTools).toEqual(["observe"]);
    expect(report.adoptionEvidence?.blockers).toEqual(["authentication_failed"]);
    expect(report.adoptionEvidence?.blockerDetails).toEqual([
      "system:error: Not logged in - Please run /login",
    ]);
    expect(report.adoptionEvidence?.mcpServerFailures).toEqual(["engram"]);
    expect(report.additionalAdoptionEvidence?.[0].client).toBe("Windsurf");
    expect(report.additionalAdoptionEvidence?.[0].blockers).toEqual([
      "mcp_server_failed",
    ]);
    expect(report.adoptionClientEvidence?.observedClients).toEqual([
      "Cursor",
      "Windsurf",
    ]);
    expect(report.adoptionClientEvidence?.blockers).toEqual(["mcp_server_failed"]);
    expect(report.adoptionClientEvidence?.mcpServerFailures).toEqual(["engram"]);
    expect(report.adoptionClientEvidence?.reports[1].blockers).toEqual([
      "mcp_server_failed",
    ]);
    expect(report.adoptionClientEvidence?.reports[1].artifactSha256).toBe("windsurf123");
    expect(report.consolidate.latestCycle).toMatchObject({
      error: "calibration failed",
      phase_issue: "calibrate: no teacher labels",
    });
  });

  it("preserves calibration needs-quality status from the backend", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        consolidate: {
          status: "attention",
          calibration: {
            status: "needs_quality",
            snapshot_count: 1,
            phase_totals: {
              triage: {
                snapshots: 1,
                total_traces: 4,
                labeled_examples: 0,
                oracle_examples: 0,
                abstain_count: 0,
                accuracy: null,
                mean_confidence: null,
                expected_calibration_error: null,
              },
            },
          },
        },
        coverage_gaps: [
          "consolidation calibration quality needs labeled decision outcomes",
        ],
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const report = await api.getEvaluationReport();

    expect(report.consolidate.status).toBe("attention");
    expect(report.consolidate.calibration.status).toBe("needs_quality");
    expect(report.evaluationSignals.falseRecall.status).toBe("needs_data");
    expect(report.consolidate.calibration.phaseTotals.triage.labeledExamples).toBe(0);
    expect(report.coverageGaps).toContain(
      "consolidation calibration quality needs labeled decision outcomes",
    );
  });
});

describe("api evaluation label writes", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("posts recall evaluation labels to the REST contract", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        status: "stored",
        groupId: "default",
        sample: {
          id: "ers_1",
          recallTriggered: true,
          recallHelped: true,
          recallNeeded: true,
          packetsSurfaced: 3,
          packetsUsed: 2,
          falseRecalls: 1,
          stalePackets: 1,
          correctedPackets: 1,
          source: "dashboard",
          query: "open loop",
          notes: null,
          timestamp: 1,
        },
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const response = await api.recordRecallEvaluation({
      recallTriggered: true,
      recallHelped: true,
      recallNeeded: true,
      packetsSurfaced: 3,
      packetsUsed: 2,
      falseRecalls: 1,
      stalePackets: 1,
      correctedPackets: 1,
      source: "dashboard",
      query: "open loop",
      notes: null,
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/evaluation/recall-samples",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          recallTriggered: true,
          recallHelped: true,
          recallNeeded: true,
          packetsSurfaced: 3,
          packetsUsed: 2,
          falseRecalls: 1,
          stalePackets: 1,
          correctedPackets: 1,
          source: "dashboard",
          query: "open loop",
          notes: null,
        }),
      }),
    );
    expect(response.sample.id).toBe("ers_1");
  });

  it("posts session continuity labels to the REST contract", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        status: "stored",
        groupId: "default",
        sample: {
          id: "esc_1",
          baselineScore: 0.2,
          memoryScore: 0.6,
          openLoopExpected: true,
          openLoopRecovered: true,
          temporalExpected: false,
          temporalCorrect: false,
          source: "dashboard",
          scenario: "follow up",
          notes: null,
          timestamp: 1,
        },
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const response = await api.recordSessionContinuityEvaluation({
      baselineScore: 0.2,
      memoryScore: 0.6,
      openLoopExpected: true,
      openLoopRecovered: true,
      temporalExpected: false,
      temporalCorrect: false,
      source: "dashboard",
      scenario: "follow up",
      notes: null,
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/evaluation/session-samples",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          baselineScore: 0.2,
          memoryScore: 0.6,
          openLoopExpected: true,
          openLoopRecovered: true,
          temporalExpected: false,
          temporalCorrect: false,
          source: "dashboard",
          scenario: "follow up",
          notes: null,
        }),
      }),
    );
    expect(response.sample.id).toBe("esc_1");
  });
});
