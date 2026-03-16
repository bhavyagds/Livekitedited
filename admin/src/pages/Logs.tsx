import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getAuditLogs, getErrorLogs } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { formatDate } from '@/lib/utils'
import {
  FileSearch,
  AlertTriangle,
  Shield,
  RefreshCw,
  Filter,
} from 'lucide-react'

type Tab = 'audit' | 'errors'

export default function Logs() {
  const [activeTab, setActiveTab] = useState<Tab>('audit')
  const [serviceFilter, setServiceFilter] = useState<string>('')
  const [levelFilter, setLevelFilter] = useState<string>('')

  const {
    data: auditLogs,
    isLoading: auditLoading,
    refetch: refetchAudit,
  } = useQuery({
    queryKey: ['audit-logs'],
    queryFn: () => getAuditLogs(100, 0),
    enabled: activeTab === 'audit',
    retry: (failureCount, error: any) => {
      // Don't retry on 401 errors (authentication issues)
      if (error?.response?.status === 401) {
        return false
      }
      return failureCount < 1
    },
  })

  const {
    data: errorLogs,
    isLoading: errorLoading,
    refetch: refetchErrors,
  } = useQuery({
    queryKey: ['error-logs', serviceFilter, levelFilter],
    queryFn: () => getErrorLogs(serviceFilter || undefined, levelFilter || undefined, 100),
    enabled: activeTab === 'errors',
    retry: (failureCount, error: any) => {
      // Don't retry on 401 errors (authentication issues)
      if (error?.response?.status === 401) {
        return false
      }
      return failureCount < 1
    },
  })

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Logs</h1>
        <p className="text-slate-500 mt-1">View audit logs and system errors.</p>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-2 border-b">
        <button
          onClick={() => setActiveTab('audit')}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            activeTab === 'audit'
              ? 'border-blue-500 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-700'
          }`}
        >
          <Shield className="w-4 h-4 inline-block mr-2" />
          Audit Logs
        </button>
        <button
          onClick={() => setActiveTab('errors')}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            activeTab === 'errors'
              ? 'border-blue-500 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-700'
          }`}
        >
          <AlertTriangle className="w-4 h-4 inline-block mr-2" />
          Error Logs
        </button>
      </div>

      {/* Audit Logs Tab */}
      {activeTab === 'audit' && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Shield className="w-5 h-5" />
                Audit Logs
              </CardTitle>
              <CardDescription>Track all admin actions</CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={() => refetchAudit()}>
              <RefreshCw className="w-4 h-4 mr-2" />
              Refresh
            </Button>
          </CardHeader>
          <CardContent>
            {auditLoading ? (
              <p className="text-slate-500 text-center py-8">Loading...</p>
            ) : auditLogs?.logs?.length === 0 ? (
              <p className="text-slate-500 text-center py-8">No audit logs yet</p>
            ) : (
              <div className="space-y-2">
                {auditLogs?.logs?.map((log: {
                  id: string
                  user_email: string
                  action: string
                  resource_type: string
                  resource_id: string
                  ip_address: string
                  created_at: string
                }) => (
                  <div
                    key={log.id}
                    className="flex items-start justify-between p-4 rounded-lg border hover:bg-slate-50"
                  >
                    <div className="flex items-start gap-4">
                      <div
                        className={`mt-1 p-1.5 rounded-full ${
                          log.action === 'login'
                            ? 'bg-green-100'
                            : log.action === 'logout'
                            ? 'bg-slate-100'
                            : log.action.includes('delete')
                            ? 'bg-red-100'
                            : 'bg-blue-100'
                        }`}
                      >
                        <Shield
                          className={`w-4 h-4 ${
                            log.action === 'login'
                              ? 'text-green-600'
                              : log.action === 'logout'
                              ? 'text-slate-600'
                              : log.action.includes('delete')
                              ? 'text-red-600'
                              : 'text-blue-600'
                          }`}
                        />
                      </div>
                      <div>
                        <p className="font-medium">{log.action.replace(/_/g, ' ')}</p>
                        <p className="text-sm text-slate-500">
                          {log.user_email}
                          {log.resource_type && (
                            <>
                              {' • '}
                              <span className="font-mono">{log.resource_type}</span>
                            </>
                          )}
                          {log.resource_id && (
                            <>
                              {' • '}
                              <span className="font-mono text-xs">{log.resource_id.slice(0, 8)}</span>
                            </>
                          )}
                        </p>
                        {log.ip_address && (
                          <p className="text-xs text-slate-400 mt-1">IP: {log.ip_address}</p>
                        )}
                      </div>
                    </div>
                    <p className="text-sm text-slate-500 whitespace-nowrap">
                      {formatDate(log.created_at)}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Error Logs Tab */}
      {activeTab === 'errors' && (
        <>
          {/* Filters */}
          <Card>
            <CardContent className="p-4">
              <div className="flex flex-wrap items-center gap-4">
                <div className="flex items-center gap-2">
                  <Filter className="w-4 h-4 text-slate-500" />
                  <span className="text-sm font-medium">Filters:</span>
                </div>
                <select
                  value={serviceFilter}
                  onChange={(e) => setServiceFilter(e.target.value)}
                  className="px-3 py-1.5 text-sm border rounded-md bg-white"
                >
                  <option value="">All Services</option>
                  <option value="agent">Agent</option>
                  <option value="api">API</option>
                  <option value="sip">SIP</option>
                  <option value="livekit">LiveKit</option>
                </select>
                <select
                  value={levelFilter}
                  onChange={(e) => setLevelFilter(e.target.value)}
                  className="px-3 py-1.5 text-sm border rounded-md bg-white"
                >
                  <option value="">All Levels</option>
                  <option value="ERROR">Error</option>
                  <option value="WARNING">Warning</option>
                  <option value="CRITICAL">Critical</option>
                </select>
                {(serviceFilter || levelFilter) && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setServiceFilter('')
                      setLevelFilter('')
                    }}
                  >
                    Clear filters
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5" />
                  Error Logs
                </CardTitle>
                <CardDescription>System errors and warnings</CardDescription>
              </div>
              <Button variant="outline" size="sm" onClick={() => refetchErrors()}>
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </Button>
            </CardHeader>
            <CardContent>
              {errorLoading ? (
                <p className="text-slate-500 text-center py-8">Loading...</p>
              ) : errorLogs?.logs?.length === 0 ? (
                <div className="text-center py-8 text-slate-500">
                  <FileSearch className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>No error logs found</p>
                  <p className="text-sm">Errors will appear here when they occur</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {errorLogs?.logs?.map((log: {
                    id: string
                    service: string
                    level: string
                    message: string
                    context: object
                    stack_trace: string
                    created_at: string
                  }) => (
                    <div
                      key={log.id}
                      className={`p-4 rounded-lg border ${
                        log.level === 'CRITICAL'
                          ? 'border-red-300 bg-red-50'
                          : log.level === 'ERROR'
                          ? 'border-orange-300 bg-orange-50'
                          : 'border-yellow-300 bg-yellow-50'
                      }`}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span
                            className={`px-2 py-0.5 text-xs font-medium rounded ${
                              log.level === 'CRITICAL'
                                ? 'bg-red-500 text-white'
                                : log.level === 'ERROR'
                                ? 'bg-orange-500 text-white'
                                : 'bg-yellow-500 text-white'
                            }`}
                          >
                            {log.level}
                          </span>
                          <span className="text-sm font-medium capitalize">{log.service}</span>
                        </div>
                        <span className="text-sm text-slate-500">
                          {formatDate(log.created_at)}
                        </span>
                      </div>
                      <p className="text-sm font-medium mb-2">{log.message}</p>
                      {log.context && Object.keys(log.context).length > 0 && (
                        <pre className="text-xs bg-white/50 p-2 rounded overflow-auto max-h-24">
                          {JSON.stringify(log.context, null, 2)}
                        </pre>
                      )}
                      {log.stack_trace && (
                        <details className="mt-2">
                          <summary className="text-xs text-slate-500 cursor-pointer">
                            Stack trace
                          </summary>
                          <pre className="text-xs bg-white/50 p-2 rounded overflow-auto max-h-48 mt-1">
                            {log.stack_trace}
                          </pre>
                        </details>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
