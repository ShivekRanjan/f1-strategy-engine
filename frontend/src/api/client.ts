// Tiny typed fetch wrapper around the FastAPI backend.
import type {
  DegradationResp,
  LapHistory,
  LiveResp,
  OutcomeResp,
  RaceInfo,
  CalendarResp,
  ConstructorIndexRow,
  ConstructorProfile,
  DriverIndexRow,
  DriverProfile,
  NewsResp,
  RaceCardResp,
  RecommendResp,
  SimulateResp,
  StandingsResp,
  TrackInfo,
  UndercutResp,
  UpcomingResp,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  // Free-tier hosting sleeps when idle and takes ~20-60s to wake, during which
  // fetches die with a network error. Every endpoint here is read-only, so
  // retrying network failures (NOT HTTP errors) with backoff rides out the
  // wake-up instead of surfacing "Failed to fetch" to the user.
  const delays = [2000, 5000, 10000];
  for (let attempt = 0; ; attempt++) {
    try {
      const res = await fetch(`${BASE}${path}`, {
        headers: { "Content-Type": "application/json" },
        ...init,
      });
      if (!res.ok) {
        let detail = `${res.status} ${res.statusText}`;
        try {
          const body = await res.json();
          if (body?.detail) detail = String(body.detail);
        } catch {
          /* non-JSON error body */
        }
        throw new Error(detail); // real API error — don't retry
      }
      return (await res.json()) as T;
    } catch (e) {
      const isNetwork = e instanceof TypeError; // fetch network failure
      if (!isNetwork || attempt >= delays.length) throw e;
      await sleep(delays[attempt]);
    }
  }
}

const post = <T>(path: string, body: unknown) =>
  req<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  tracks: () => req<{ tracks: string[] }>("/tracks"),
  tracksInfo: () => req<{ tracks: TrackInfo[] }>("/tracks_info"),
  raceInfo: (track: string) => req<RaceInfo>(`/race/${encodeURIComponent(track)}`),
  degradation: (track: string, season: number | null, cliff: boolean) =>
    req<DegradationResp>(
      `/degradation/${encodeURIComponent(track)}?cliff=${cliff}` +
        (season != null ? `&season=${season}` : ""),
    ),
  seasons: (track: string) =>
    req<{ seasons: number[] }>(`/seasons/${encodeURIComponent(track)}`),
  allSeasons: () => req<{ seasons: number[] }>("/seasons"),
  circuits: (season: number) =>
    req<{ season: number; circuits: string[] }>(`/circuits/${season}`),
  drivers: (track: string, season: number) =>
    req<{ drivers: string[] }>(`/drivers/${encodeURIComponent(track)}/${season}`),
  laps: (track: string, season: number, driver: string) =>
    req<LapHistory>(
      `/laps/${encodeURIComponent(track)}/${season}/${encodeURIComponent(driver)}`,
    ),

  recommend: (body: {
    track: string;
    objective?: string;
    use_cliff?: boolean;
    max_stops?: number;
    n_runs?: number;
    top_k?: number;
    season?: number | null;
    sc_scale?: number;
    track_temp?: number | null;
  }) => post<RecommendResp>("/recommend", body),

  simulate: (body: {
    track: string;
    compounds: string[];
    pit_laps: number[];
    use_cliff?: boolean;
    n_runs?: number;
    season?: number | null;
  }) => post<SimulateResp>("/simulate", body),

  undercut: (body: {
    track: string;
    current_lap: number;
    gap_s: number;
    your_compound: string;
    your_age: number;
    your_new_compound: string;
    rival_compound: string;
    rival_age: number;
    rival_new_compound: string;
    rival_pit_lap: number;
    season?: number | null;
    n_runs?: number;
  }) => post<UndercutResp>("/undercut", body),

  live: (body: {
    track: string;
    season: number;
    driver: string;
    current_lap: number;
    n_runs?: number;
  }) => post<LiveResp>("/live", body),

  outcome: () => req<OutcomeResp>("/outcome"),
  predictUpcoming: (grid?: Record<string, number>) =>
    post<UpcomingResp>("/predict_upcoming", { grid: grid ?? null }),
  standings: (season?: number) =>
    req<StandingsResp>(`/standings${season != null ? `?season=${season}` : ""}`),
  standingsRefresh: (season?: number) =>
    req<StandingsResp>(`/standings/refresh${season != null ? `?season=${season}` : ""}`),
  raceCard: (season: number, track: string) =>
    req<RaceCardResp>(`/race_card/${season}/${encodeURIComponent(track)}`),
  driversIndex: () => req<{ drivers: DriverIndexRow[] }>("/profiles/drivers"),
  constructorsIndex: () => req<{ constructors: ConstructorIndexRow[] }>("/profiles/constructors"),
  driverProfile: (code: string) =>
    req<DriverProfile>(`/profiles/driver/${encodeURIComponent(code)}`),
  constructorProfile: (team: string) =>
    req<ConstructorProfile>(`/profiles/constructor/${encodeURIComponent(team)}`),
  news: (limit = 40) => req<NewsResp>(`/news?limit=${limit}`),
  calendar: (season: number) => req<CalendarResp>(`/calendar/${season}`),
};
