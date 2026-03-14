import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"
import { format as formatDateFns, formatDistanceToNow } from "date-fns"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(date: string | Date): string {
  const dateObj = typeof date === "string" ? new Date(date) : date
  return formatDateFns(dateObj, "MMM d, yyyy 'at' h:mm a")
}

export function formatDuration(seconds: number | null | undefined): string {
  if (!seconds) return "0s"
  
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = seconds % 60
  
  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`
  } else {
    return `${secs}s`
  }
}

export function formatRelativeTime(date: string | Date): string {
  const dateObj = typeof date === "string" ? new Date(date) : date
  return formatDistanceToNow(dateObj, { addSuffix: true })
}
