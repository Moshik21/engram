import { useEffect, useState } from "react";

export type ShowcaseStep = {
  label: string;
  detail: string;
};

export type ShowcaseBeat = {
  id: string;
  title: string;
  user_message: string;
  action: string;
  query: string;
  narrative: string;
  answer_hint: string;
  passed: boolean;
  matched_tokens: string[];
  highlights: string[];
  steps: ShowcaseStep[];
};

export type ShowcaseExport = {
  kind: string;
  version: number;
  group_id: string;
  beats: ShowcaseBeat[];
  summary: {
    passed: number;
    total: number;
  };
};

export function useShowcaseExport(path = "/showcase-export.json") {
  const [data, setData] = useState<ShowcaseExport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(path)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load showcase export (${response.status})`);
        }
        return response.json() as Promise<ShowcaseExport>;
      })
      .then((payload) => {
        if (!cancelled) {
          setData(payload);
          setError(null);
        }
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setData(null);
          setError(err.message);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [path]);

  return { data, error };
}