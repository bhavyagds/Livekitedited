import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getAnalytics } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Phone,
  CheckCircle,
  XCircle,
  Clock,
} from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
} from 'recharts'

export default function Analytics() {
  const [days, setDays] = useState(30)

  const { data, isLoading } = useQuery({
    queryKey: ['analytics', days],
    queryFn: () => getAnalytics(days),
    retry: (failureCount, error: any) => {
      // Don't retry on 401 errors (authentication issues)
      if (error?.response?.status === 401) {
        return false
      }
      return failureCount < 1
    },
  })

  const summary = data?.summary || {}
  const dailyData = data?.summary?.daily_data || []

  // Reverse to show oldest first
  const chartData = [...dailyData].reverse().map((d: {
    date: string
    total_calls: number
    successful_calls: number
    failed_calls: number
  }) => ({
    date: new Date(d.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    total: d.total_calls,
    successful: d.successful_calls,
    failed: d.failed_calls,
  }))

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Analytics</h1>
          <p className="text-slate-500 mt-1">Call performance metrics and trends.</p>
        </div>
        <div className="flex gap-2">
          {[7, 14, 30, 60].map((d) => (
            <Button
              key={d}
              variant={days === d ? 'default' : 'outline'}
              size="sm"
              onClick={() => setDays(d)}
            >
              {d} days
            </Button>
          ))}
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-500">Total Calls</p>
                <p className="text-3xl font-bold">{summary.total_calls || 0}</p>
              </div>
              <div className="p-3 bg-blue-100 rounded-full">
                <Phone className="w-6 h-6 text-blue-600" />
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-500">Successful</p>
                <p className="text-3xl font-bold text-green-600">
                  {summary.successful_calls || 0}
                </p>
              </div>
              <div className="p-3 bg-green-100 rounded-full">
                <CheckCircle className="w-6 h-6 text-green-600" />
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-500">Failed</p>
                <p className="text-3xl font-bold text-red-600">{summary.failed_calls || 0}</p>
              </div>
              <div className="p-3 bg-red-100 rounded-full">
                <XCircle className="w-6 h-6 text-red-600" />
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-500">Success Rate</p>
                <p className="text-3xl font-bold">
                  {summary.success_rate?.toFixed(1) || 0}%
                </p>
              </div>
              <div
                className={`p-3 rounded-full ${
                  (summary.success_rate || 0) >= 80 ? 'bg-green-100' : 'bg-yellow-100'
                }`}
              >
                {(summary.success_rate || 0) >= 80 ? (
                  <TrendingUp className="w-6 h-6 text-green-600" />
                ) : (
                  <TrendingDown className="w-6 h-6 text-yellow-600" />
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Daily Call Volume
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="h-64 flex items-center justify-center text-slate-500">
                Loading...
              </div>
            ) : chartData.length === 0 ? (
              <div className="h-64 flex items-center justify-center text-slate-500">
                No data available
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="successful" fill="#22c55e" name="Successful" stackId="a" />
                  <Bar dataKey="failed" fill="#ef4444" name="Failed" stackId="a" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              Call Trends
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="h-64 flex items-center justify-center text-slate-500">
                Loading...
              </div>
            ) : chartData.length === 0 ? (
              <div className="h-64 flex items-center justify-center text-slate-500">
                No data available
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="total"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    name="Total Calls"
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Average Duration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="w-5 h-5" />
            Performance Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center p-4 bg-slate-50 rounded-lg">
              <p className="text-3xl font-bold text-blue-600">
                {Math.round(summary.avg_duration_seconds || 0)}s
              </p>
              <p className="text-sm text-slate-500 mt-1">Avg Duration</p>
            </div>
            <div className="text-center p-4 bg-slate-50 rounded-lg">
              <p className="text-3xl font-bold text-green-600">
                {Math.round((summary.total_calls || 0) / days)}
              </p>
              <p className="text-sm text-slate-500 mt-1">Avg Calls/Day</p>
            </div>
            <div className="text-center p-4 bg-slate-50 rounded-lg">
              <p className="text-3xl font-bold text-purple-600">
                {summary.successful_calls || 0}
              </p>
              <p className="text-sm text-slate-500 mt-1">Handled</p>
            </div>
            <div className="text-center p-4 bg-slate-50 rounded-lg">
              <p className="text-3xl font-bold text-orange-600">{days}</p>
              <p className="text-sm text-slate-500 mt-1">Days Analyzed</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
