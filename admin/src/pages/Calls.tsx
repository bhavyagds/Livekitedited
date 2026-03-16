import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getCalls } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { formatDate, formatDuration } from '@/lib/utils'
import {
  Phone,
  PhoneIncoming,
  PhoneOutgoing,
  Monitor,
  ChevronLeft,
  ChevronRight,
  Filter,
} from 'lucide-react'

export default function Calls() {
  const [page, setPage] = useState(1)
  const [statusFilter, setStatusFilter] = useState<string | undefined>()
  const [typeFilter, setTypeFilter] = useState<string | undefined>()
  const pageSize = 20

  const { data, isLoading } = useQuery({
    queryKey: ['calls', page, statusFilter, typeFilter],
    queryFn: () => getCalls(page, pageSize, statusFilter, typeFilter),
    retry: (failureCount, error: any) => {
      // Don't retry on 401 errors (authentication issues)
      if (error?.response?.status === 401) {
        return false
      }
      return failureCount < 1
    },
  })

  const totalPages = data ? Math.ceil(data.total / pageSize) : 0

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Call History</h1>
        <p className="text-slate-500 mt-1">View and analyze all calls.</p>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-slate-500" />
              <span className="text-sm font-medium">Filters:</span>
            </div>
            <div className="flex flex-wrap gap-2">
              <select
                value={statusFilter || ''}
                onChange={(e) => {
                  setStatusFilter(e.target.value || undefined)
                  setPage(1)
                }}
                className="px-3 py-1.5 text-sm border rounded-md bg-white"
              >
                <option value="">All Status</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
                <option value="missed">Missed</option>
                <option value="active">Active</option>
              </select>
              <select
                value={typeFilter || ''}
                onChange={(e) => {
                  setTypeFilter(e.target.value || undefined)
                  setPage(1)
                }}
                className="px-3 py-1.5 text-sm border rounded-md bg-white"
              >
                <option value="">All Types</option>
                <option value="web">Web</option>
                <option value="inbound">Inbound (SIP)</option>
                <option value="outbound">Outbound (SIP)</option>
              </select>
            </div>
            {(statusFilter || typeFilter) && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setStatusFilter(undefined)
                  setTypeFilter(undefined)
                  setPage(1)
                }}
              >
                Clear filters
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Calls Table */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Phone className="w-5 h-5" />
            Calls ({data?.total || 0})
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <p className="text-slate-500 text-center py-8">Loading...</p>
          ) : data?.calls?.length === 0 ? (
            <p className="text-slate-500 text-center py-8">No calls found</p>
          ) : (
            <>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-3 px-4 text-sm font-medium text-slate-500">Type</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-slate-500">Caller</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-slate-500">Status</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-slate-500">Duration</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-slate-500">Started</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-slate-500">Room</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data?.calls?.map((call: {
                      id: string
                      call_type: string
                      caller_number: string
                      caller_name: string
                      status: string
                      duration_seconds: number
                      started_at: string
                      room_name: string
                      disconnect_reason: string
                    }) => (
                      <tr key={call.id} className="border-b hover:bg-slate-50">
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2">
                            {call.call_type === 'web' ? (
                              <Monitor className="w-4 h-4 text-purple-500" />
                            ) : call.call_type === 'inbound' ? (
                              <PhoneIncoming className="w-4 h-4 text-green-500" />
                            ) : (
                              <PhoneOutgoing className="w-4 h-4 text-blue-500" />
                            )}
                            <span className="text-sm capitalize">{call.call_type}</span>
                          </div>
                        </td>
                        <td className="py-3 px-4">
                          <div>
                            <p className="font-medium text-sm">
                              {call.caller_number || 'Web User'}
                            </p>
                            {call.caller_name && (
                              <p className="text-xs text-slate-500">{call.caller_name}</p>
                            )}
                          </div>
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={`inline-block px-2 py-1 text-xs rounded-full ${
                              call.status === 'completed'
                                ? 'bg-green-100 text-green-700'
                                : call.status === 'failed'
                                ? 'bg-red-100 text-red-700'
                                : call.status === 'active'
                                ? 'bg-blue-100 text-blue-700'
                                : 'bg-yellow-100 text-yellow-700'
                            }`}
                          >
                            {call.status}
                          </span>
                          {call.disconnect_reason && call.disconnect_reason !== 'normal' && (
                            <p className="text-xs text-red-500 mt-0.5">{call.disconnect_reason}</p>
                          )}
                        </td>
                        <td className="py-3 px-4 text-sm">
                          {formatDuration(call.duration_seconds || 0)}
                        </td>
                        <td className="py-3 px-4 text-sm text-slate-500">
                          {formatDate(call.started_at)}
                        </td>
                        <td className="py-3 px-4 text-sm text-slate-500 font-mono">
                          {call.room_name ? call.room_name.slice(0, 16) + '...' : '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex items-center justify-between mt-4 pt-4 border-t">
                  <p className="text-sm text-slate-500">
                    Page {page} of {totalPages} ({data?.total} total calls)
                  </p>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage((p) => Math.max(1, p - 1))}
                      disabled={page === 1}
                    >
                      <ChevronLeft className="w-4 h-4" />
                      Previous
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                      disabled={page === totalPages}
                    >
                      Next
                      <ChevronRight className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
