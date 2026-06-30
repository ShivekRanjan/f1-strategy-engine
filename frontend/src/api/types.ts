// Response shapes from the FastAPI backend (f1se.api). Only the fields the UI
// reads are typed; the engine returns a few extras we ignore.

export interface RaceInfo {
  track: string;
  total_laps: number;
  sc_prob_per_lap: number;
  sc_mean_duration: number;
  pit_loss_s: number;
  stint_limits: Record<string, number>;
  well_sampled: boolean;
}

export interface StrategySummary {
  compounds: string[];
  pit_laps: number[];
  mean_s: number;
  p50_s: number;
  p90_s: number;
}

export interface ShortlistRow {
  rank: number;
  compounds: string[];
  pit_laps: number[];
  mean_s: number;
  p50_s: number;
  p90_s: number;
  win_prob_vs_best: number;
}

export interface RecommendResp extends RaceInfo {
  objective: string;
  use_cliff: boolean;
  n_evaluated: number;
  best: StrategySummary;
  shortlist: ShortlistRow[];
}

export interface SimulateResp {
  mean_s: number;
  p10_s: number;
  p50_s: number;
  p90_s: number;
  p_safety_car: number;
  hist_counts: number[];
  hist_edges: number[];
}

export interface UndercutOption {
  final_gap_s: number;
  p_ahead: number;
}
export interface UndercutResp {
  verdict: string;
  undercut_works: boolean;
  undercut_gain_s: number;
  undercut: UndercutOption;
  cover: UndercutOption;
}

export interface LapRow {
  lap: number;
  stint: number | null;
  compound: string;
  tyre_age: number | null;
  lap_time_s: number | null;
  lap_time_fuel_corr_s: number | null;
  position: number | null;
}
export interface LapHistory {
  track: string;
  season: number;
  driver: string;
  total_laps: number;
  lap_min: number;
  lap_max: number;
  laps: LapRow[];
}

export interface LiveShortlistRow {
  rank: number;
  plan: string;
  mean_remaining_s: number;
  win_prob_vs_best: number;
}
export interface LiveRecommendation {
  best_plan: string;
  laps_remaining: number;
  n_evaluated: number;
  shortlist: LiveShortlistRow[];
}
export interface Nowcast {
  ok: boolean;
  reason?: string;
  last_s?: number;
  predicted_s?: number;
  delta_s?: number;
  window?: number;
}
export interface LiveState {
  current_lap: number;
  total_laps: number;
  current_compound: string;
  tyre_age: number;
  laps_remaining: number;
  compounds_used: string[];
}
export interface LiveResp {
  track: string;
  season: number;
  driver: string;
  state: LiveState;
  recommendation: LiveRecommendation | null;
  rec_note: string | null;
  nowcast: Nowcast | null;
}

export interface ChampRow {
  driver: string;
  win_prob: number;
  points: number | null;
}
export interface PodiumPred {
  driver: string;
  team: string;
  grid: number;
  podium_prob: number;
  actual: boolean;
}
export interface OutcomeRound {
  round: number;
  event_name: string;
  predictions: PodiumPred[];
}
export interface OutcomeResp {
  test_year: number;
  ongoing: boolean;
  done: number;
  full: number;
  championship: ChampRow[];
  model_metrics: {
    auc: number;
    model_precision_at_3: number;
    grid_baseline_precision_at_3: number;
  };
  rounds: OutcomeRound[];
}
