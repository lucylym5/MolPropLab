import React from "react";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "ghost" };
export const Button: React.FC<ButtonProps> = ({ className = "", variant = "primary", ...props }) => {
  const base = "px-4 py-2 rounded-md font-medium transition";
  const styles =
    variant === "primary"
      ? "bg-primary text-white hover:opacity-90"
      : "bg-transparent border border-border text-foreground hover:bg-card";
  return <button className={`${base} ${styles} ${className}`} {...props} />;
};

export const Card: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ className = "", ...props }) => (
  <div className={`rounded-xl border border-border bg-card p-5 shadow-md ${className}`} {...props} />
);

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className = "", ...props }, ref) => (
    <input
      ref={ref}
      className={`w-full rounded-md bg-transparent border border-border px-3 py-2 outline-none text-foreground placeholder:opacity-60 ${className}`}
      {...props}
    />
  )
);
Input.displayName = "Input";

export const Textarea = React.forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(
  ({ className = "", ...props }, ref) => (
    <textarea
      ref={ref}
      className={`w-full rounded-md bg-white border border-border px-3 py-2 outline-none text-foreground placeholder:opacity-60 ${className}`}
      {...props}
    />
  )
);
Textarea.displayName = "Textarea";

export const Badge: React.FC<React.HTMLAttributes<HTMLSpanElement> & { color?: string }> = ({ className = "", color, ...props }) => (
  <span
    className={`inline-block rounded-full px-2 py-1 text-xs font-semibold ${className}`}
    style={{ background: color ?? "rgba(59,130,246,0.15)", color: "#1e40af" }}
    {...props}
  />
);

export const Progress: React.FC<{ value: number }> = ({ value }) => (
  <div className="h-2 w-full rounded bg-gray-200">
    <div className="h-2 rounded bg-accent" style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
  </div>
);

export const Table: React.FC<{ columns: string[]; rows: (string | number)[][] }> = ({ columns, rows }) => (
  <div className="overflow-x-auto">
    <table className="w-full text-sm">
      <thead className="text-left text-muted border-b border-border">
        <tr>{columns.map((c) => <th key={c} className="px-2 py-2">{c}</th>)}</tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i} className="border-b border-border/50">
            {r.map((v, j) => <td key={j} className="px-2 py-2">{String(v)}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);
