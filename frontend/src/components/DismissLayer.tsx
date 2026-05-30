type Props = {
  onDismiss: () => void
}

/** Transparent full-screen click target — must not be a `<button>` (avoids global button hover styles). */
export function DismissLayer({ onDismiss }: Props) {
  return <div className="dismiss-layer" onClick={onDismiss} aria-hidden="true" />
}
