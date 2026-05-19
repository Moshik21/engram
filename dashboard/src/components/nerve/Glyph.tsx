type GlyphProps = {
  label: string;
  size?: number;
  className?: string;
  title?: string;
};

export function Glyph({ label, size = 18, className = "", title }: GlyphProps) {
  return (
    <span
      aria-hidden={title ? undefined : true}
      aria-label={title}
      className={`inline-flex items-center justify-center rounded-md font-mono font-bold leading-none ${className}`}
      style={{ width: size, height: size, fontSize: Math.max(9, Math.floor(size * 0.48)) }}
    >
      {label}
    </span>
  );
}
