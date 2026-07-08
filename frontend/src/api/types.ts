// Response shapes from the FastAPI backend (f1se.api). Only the fields the UI
// reads are typed; the engine returns a few extras we ignore.

export interface TrackInfo {
  track: string;
  total_laps: number;
  well_sampled: boolean;
}

export interface CompoundCurve {
  max_age: number;
  onset: number | null;
  slope: number;
  ages: number[];
  linear: number[];
  cliff: number[];
}
export interface DegradationResp {
  track: string;
  season: number | null;
  use_cliff: boolean;
  compounds: Record<string, CompoundCurve>;
}

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

export interface UpcomingPred {
  driver: string;
  team: string;
  grid: number;
  podium_prob: number;
}
export interface UpcomingResp {
  season: number;
  next_round: number;
  grid_source: string;
  predictions: UpcomingPred[];
}

export interface DriverStanding {
  pos: number;
  driver: string;
  team: string;
  points: number;
  wins: number;
  podiums: number;
  races: number;
  win_prob?: number | null;
}
export interface ConstructorStanding {
  pos: number;
  team: string;
  points: number;
  wins: number;
  podiums: number;
}
export interface StandingsResp {
  season: number;
  seasons: number[];
  latest: number;
  races_done: number;
  total_races: number;
  ongoing: boolean;
  includes_sprints?: boolean;
  drivers: DriverStanding[];
  constructors: ConstructorStanding[];
  refreshed?: boolean;
  added_rounds?: number[];
  as_of?: string;
}

export interface RaceResultRow {
  pos: number | null;
  driver: string;
  team: string;
  grid: number | null;
  points: number | null;
  status: string;
  gained: number | null;
}
export interface RaceCardPred {
  driver: string;
  team: string;
  grid: number;
  podium_prob: number;
  actual: boolean;
}
export interface RaceCardPrediction {
  predictions: RaceCardPred[];
  hit_at_3: number;
  auc: number;
}
export interface RaceCardResp {
  season: number;
  round: number;
  event_name: string;
  result: RaceResultRow[];
  actual_podium: string[];
  prediction: RaceCardPrediction | null;
}

export interface DriverIndexRow {
  driver: string;
  team: string;
  last_season: number;
  seasons: number[];
  points: number | null;
  wins: number;
}
export interface ConstructorIndexRow {
  team: string;
  last_season: number;
  seasons: number[];
  points: number | null;
  wins: number;
}
export interface DriverSeasonLine {
  season: number;
  team: string;
  races: number;
  wins: number;
  podiums: number;
  points: number | null;
  avg_grid: number | null;
  avg_finish: number | null;
  best: number | null;
  dnf: number;
}
export interface RecentResult {
  season: number;
  round: number;
  event_name: string;
  grid: number | null;
  position: number | null;
  points: number | null;
  status: string;
}
export interface TeammateH2H {
  teammate: string;
  quali_races: number;
  quali_ahead: number;
  race_races: number;
  race_ahead: number;
  pts_self: number | null;
  pts_mate: number | null;
}
export interface DriverProfile {
  driver: string;
  team: string;
  seasons: number[];
  career: {
    races: number;
    wins: number;
    podiums: number;
    points: number | null;
    avg_grid: number | null;
    avg_finish: number | null;
    best: number | null;
    dnf: number;
  };
  by_season: DriverSeasonLine[];
  recent: RecentResult[];
  h2h_season: number;
  teammate_h2h: TeammateH2H[];
}
export interface ConstructorSeasonLine {
  season: number;
  races: number;
  wins: number;
  podiums: number;
  points: number | null;
  best: number | null;
  drivers: string[];
}
export interface ConstructorDriver {
  driver: string;
  points: number | null;
  wins: number;
  seasons: number[];
}
export interface ConstructorProfile {
  team: string;
  seasons: number[];
  career: { races: number; wins: number; podiums: number; points: number | null; best: number | null };
  by_season: ConstructorSeasonLine[];
  drivers: ConstructorDriver[];
}

export interface NewsItem {
  title: string;
  link: string;
  source: string;
  ts: number | null;
  summary: string;
}
export interface NewsResp {
  items: NewsItem[];
  sources: string[];
  fetched_at: number;
  cached?: boolean;
}

export interface CalendarSession {
  name: string;
  date: string;
}
export interface CalendarRound {
  round: number;
  event_name: string;
  country: string;
  location: string;
  event_date: string | null;
  format: string;
  sessions: CalendarSession[];
  done: boolean;
}
export interface CalendarResp {
  season: number;
  rounds: CalendarRound[];
  next_round: number | null;
  next_session: { round: number; event_name: string; name: string; date: string } | null;
}
