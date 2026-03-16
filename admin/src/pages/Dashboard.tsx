import { useQuery } from '@tanstack/react-query'
import { getAnalytics, getHealth, getCalls } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { formatDate, formatDuration } from '@/lib/utils'
import {
  Phone,
  PhoneIncoming,
  PhoneOutgoing,
  Monitor,
  CheckCircle,
  XCircle,
  Clock,
  TrendingUp,
  Activity,
} from 'lucide-react'

export default function Dashboard() {
  const { data: analytics, isLoading: analyticsLoading } = useQuery({
    queryKey: ['analytics'],
    queryFn: () => getAnalytics(30),
    refetchInterval: (query) => {
      // Only refetch if tab is visible
      return document.visibilityState === 'visible' ? 60000 : false
    },
    retry: (failureCount, error: any) => {
      // Don't retry on 401 errors (authentication issues)
      if (error?.response?.status === 401) {
        return false
      }
      return failureCount < 1
    },
  })

  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: (query) => {
      // Only refetch if tab is visible
      return document.visibilityState === 'visible' ? 30000 : false
    },
    retry: (failureCount, error: any) => {
      // Don't retry on 401 errors (authentication issues)
      if (error?.response?.status === 401) {
        return false
      }
      return failureCount < 1
    },
  })

  const { data: recentCalls, isLoading: callsLoading } = useQuery({
    queryKey: ['recent-calls'],
    retry: (failureCount, error: any) => {
      // Don't retry on 401 errors (authentication issues)
      if (error?.response?.status === 401) {
        return false
      }
      return failureCount < 1
    },
    queryFn: () => getCalls(1, 5),
    refetchInterval: (query) => {
      // Only refetch if tab is visible
      return document.visibilityState === 'visible' ? 30000 : false
    },
  })

  const today = analytics?.today || {}
  const summary = analytics?.summary || {}

  const stats = [
    {
      name: 'Total Calls Today',
      value: today.total_calls || 0,
      icon: Phone,
      color: 'bg-blue-500',
    },
    {
      name: 'Successful',
      value: today.successful_calls || 0,
      icon: CheckCircle,
      color: 'bg-green-500',
    },
    {
      name: 'Failed',
      value: today.failed_calls || 0,
      icon: XCircle,
      color: 'bg-red-500',
    },
    {
      name: 'Web Calls',
      value: today.web_calls || 0,
      icon: Monitor,
      color: 'bg-purple-500',
    },
    {
      name: 'SIP Calls',
      value: today.sip_calls || 0,
      icon: PhoneIncoming,
      color: 'bg-orange-500',
    },
    {
      name: 'Avg Duration',
      value: formatDuration(Math.round(summary.avg_duration_seconds || 0)),
      icon: Clock,
      color: 'bg-cyan-500',
    },
  ]

  const services = health?.services || {}

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Dashboard</h1>
        <p className="text-slate-500 mt-1">
          Welcome back! Here's an overview of your call center.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {stats.map((stat) => (
          <Card key={stat.name}>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${stat.color}`}>
                  <stat.icon className="w-5 h-5 text-white" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{stat.value}</p>
                  <p className="text-xs text-slate-500">{stat.name}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Service Health */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Service Health
            </CardTitle>
          </CardHeader>
          <CardContent>
            {healthLoading ? (
              <p className="text-slate-500">Loading...</p>
            ) : (
              <div className="space-y-3">
                {Object.entries(services).map(([name, status]: [string, unknown]) => {
                  const serviceStatus = status as { configured: boolean }
                  return (
                    <div key={name} className="flex items-center justify-between py-2 border-b last:border-0">
                      <div className="flex items-center gap-2">
                        <div
                          className={`w-2 h-2 rounded-full ${
                            serviceStatus.configured ? 'bg-green-500' : 'bg-yellow-500'
                          }`}
                        />
                        <span className="font-medium capitalize">{name}</span>
                      </div>
                      <span
                        className={`text-sm ${
                          serviceStatus.configured ? 'text-green-600' : 'text-yellow-600'
                        }`}
                      >
                        {serviceStatus.configured ? 'Configured' : 'Not Configured'}
                      </span>
                    </div>
                  )
                })}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Calls */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Phone className="w-5 h-5" />
              Recent Calls
            </CardTitle>
          </CardHeader>
          <CardContent>
            {callsLoading ? (
              <p className="text-slate-500">Loading...</p>
            ) : recentCalls?.calls?.length === 0 ? (
              <p className="text-slate-500 text-center py-4">No calls yet</p>
            ) : (
              <div className="space-y-3">
                {recentCalls?.calls?.map((call: {
                  id: string
                  call_type: string
                  caller_number: string
                  status: string
                  started_at: string
                  duration_seconds: number
                }) => (
                  <div key={call.id} className="flex items-center justify-between py-2 border-b last:border-0">
                    <div className="flex items-center gap-3">
                      {call.call_type === 'web' ? (
                        <Monitor className="w-4 h-4 text-purple-500" />
                      ) : call.call_type === 'inbound' ? (
                        <PhoneIncoming className="w-4 h-4 text-green-500" />
                      ) : (
                        <PhoneOutgoing className="w-4 h-4 text-blue-500" />
                      )}
                      <div>
                        <p className="font-medium text-sm">{call.caller_number || 'Web Call'}</p>
                        <p className="text-xs text-slate-500">{formatDate(call.started_at)}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <span
                        className={`inline-block px-2 py-0.5 text-xs rounded-full ${
                          call.status === 'completed'
                            ? 'bg-green-100 text-green-700'
                            : call.status === 'failed'
                            ? 'bg-red-100 text-red-700'
                            : 'bg-yellow-100 text-yellow-700'
                        }`}
                      >
                        {call.status}
                      </span>
                      {call.duration_seconds > 0 && (
                        <p className="text-xs text-slate-500 mt-0.5">
                          {formatDuration(call.duration_seconds)}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* 30-Day Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            30-Day Summary
          </CardTitle>
        </CardHeader>
        <CardContent>
          {analyticsLoading ? (
            <p className="text-slate-500">Loading...</p>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <p className="text-3xl font-bold">{summary.total_calls || 0}</p>
                <p className="text-sm text-slate-500">Total Calls</p>
              </div>
              <div>
                <p className="text-3xl font-bold text-green-600">{summary.successful_calls || 0}</p>
                <p className="text-sm text-slate-500">Successful</p>
              </div>
              <div>
                <p className="text-3xl font-bold text-red-600">{summary.failed_calls || 0}</p>
                <p className="text-sm text-slate-500">Failed</p>
              </div>
              <div>
                <p className="text-3xl font-bold text-blue-600">
                  {summary.success_rate?.toFixed(1) || 0}%
                </p>
                <p className="text-sm text-slate-500">Success Rate</p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
