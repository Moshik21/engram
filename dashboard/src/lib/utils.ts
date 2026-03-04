export function formatRelativeTime(iso: string | null): string {
  if (!iso) return "never";
  const now = Date.now();
  // Ensure UTC interpretation: append Z if no timezone indicator present
  const normalized = /[Z+\-]\d{0,2}:?\d{0,2}$/.test(iso) ? iso : iso + "Z";
  const then = new Date(normalized).getTime();
  if (isNaN(then)) return "—";
  const diff = now - then;
  const seconds = Math.floor(diff / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  return `${months}mo ago`;
}

export function debounce<T extends (...args: Parameters<T>) => void>(
  fn: T,
  ms: number,
): (...args: Parameters<T>) => void {
  let timer: ReturnType<typeof setTimeout>;
  return (...args: Parameters<T>) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}
