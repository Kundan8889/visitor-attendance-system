        const dashboardState = {
          cameras: [],
          attendance: [],
          filteredAttendance: [],
          employeeMap: {},
          roster: [],
          auditTrail: [],
          manualMarks: [],
          lastAutoPassAt: Date.now(),
          cameraSearch: '',
          cameraFilter: 'all',
          attendanceSearch: '',
          attendanceLimit: 100,
          selectedDate: '',
          deptFilter: 'all',
          shiftFilter: 'all',
          statusFilter: 'all',
          lastSyncAt: 0,
        };
        let refreshInFlight = false;
        const openedVisitorPasses = new Set();
        function escapeHtml(v) {
          const s = String(v ?? '');
          return s
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/\"/g, '&quot;')
            .replace(/'/g, '&#39;');
        }

        function cameraCardId(cameraId) {
          return `cam-${cameraId}`;
        }

        function showToast(text, tone = '') {
          const stack = document.getElementById('toastStack');
          if (!stack) return;
          const toast = document.createElement('div');
          toast.className = `toast ${tone}`.trim();
          toast.textContent = text;
          stack.appendChild(toast);
          setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(8px)';
            toast.style.transition = 'all 0.2s ease';
            setTimeout(() => toast.remove(), 220);
          }, 2400);
        }

        function setSyncBadge(tone, text) {
          const badge = document.getElementById('syncBadge');
          if (!badge) return;
          badge.className = 'sync-status';
          if (tone) badge.classList.add(tone);
          badge.textContent = text;
        }

        function updateSyncStaleness() {
          if (!dashboardState.lastSyncAt) return;
          const ageSeconds = Math.max(0, Math.floor((Date.now() - dashboardState.lastSyncAt) / 1000));
          if (ageSeconds <= 4) setSyncBadge('good', `Live - ${ageSeconds}s ago`);
          else if (ageSeconds <= 10) setSyncBadge('warn', `Delayed - ${ageSeconds}s ago`);
          else setSyncBadge('bad', `Stale - ${ageSeconds}s ago`);
        }

        function setFormMessage(text, tone) {
          const msg = document.getElementById('formMsg');
          if (!msg) return;
          msg.textContent = text;
          msg.classList.remove('muted');
          msg.style.color = '';
          if (tone === 'good') msg.style.color = '#c2f4de';
          else if (tone === 'bad') msg.style.color = '#ffd6e1';
          else if (tone === 'warn') msg.style.color = '#ffe9c2';
          else msg.classList.add('muted');
        }

        function todayIsoDate() {
          const d = new Date();
          const m = String(d.getMonth() + 1).padStart(2, '0');
          const day = String(d.getDate()).padStart(2, '0');
          return `${d.getFullYear()}-${m}-${day}`;
        }

        function activateNav(btn, targetId) {
          const nav = document.getElementById('appNav');
          if (!nav) return;
          nav.querySelectorAll('button').forEach((node) => node.classList.remove('active'));
          if (btn) btn.classList.add('active');
          const el = document.getElementById(targetId);
          if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        function addAudit(message, level) {
          dashboardState.auditTrail.unshift({
            at: new Date().toISOString(),
            level: level || 'info',
            message: message,
          });
          dashboardState.auditTrail = dashboardState.auditTrail.slice(0, 40);
          renderAuditTrail();
        }

        function quickAddEmployee() {
          const name = (prompt('Employee name:') || '').trim();
          if (!name) return;
          const source = (prompt('Camera source (0 or RTSP URL):', '0') || '0').trim();
          fetch('/api/employees/capture', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({ name: name, source: source }),
          })
            .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
            .then(({ ok, data }) => {
              if (!ok || !data.ok) {
                showToast(data.error || 'Failed to start capture', 'bad');
                addAudit(`Capture start failed: ${name}`, 'error');
                return;
              }
              const displayName = (data.employee && data.employee.name) ? data.employee.name : name;
              showToast(`Capture started for ${displayName}`, 'good');
              addAudit(`Capture started for ${displayName}`, 'info');
            })
            .catch(() => {
              showToast('Failed to start capture', 'bad');
              addAudit(`Capture start failed: ${name}`, 'error');
            });
        }

        function parseCreatedAt(raw) {
          if (!raw) return 0;
          const t = Date.parse(raw);
          if (!Number.isNaN(t)) return t;
          return 0;
        }

        async function checkAutoVisitorPasses() {
          const sinceIso = new Date(dashboardState.lastAutoPassAt).toISOString();
          try {
            const res = await fetch(`/api/visitors/auto?since=${encodeURIComponent(sinceIso)}`);
            if (!res.ok) return;
            const passes = await res.json();
            if (!Array.isArray(passes) || !passes.length) return;
            let maxSeen = dashboardState.lastAutoPassAt;
            passes.forEach((pass) => {
              if (String(pass.name || '').trim().toLowerCase() === 'unknown') {
                return;
              }
              const passId = String(pass.id || '').trim();
              const eventType = String(pass.event || 'entry').toLowerCase();
              const eventKey = passId ? `${passId}:${eventType}` : '';
              if (eventKey && openedVisitorPasses.has(eventKey)) {
                return;
              }
              const createdAt = parseCreatedAt(pass.event_at || pass.created_at);
              if (createdAt > maxSeen) maxSeen = createdAt;
              const passUrl = pass.pass_url || `/visitors/pass/${encodeURIComponent(pass.id || '')}`;
              const win = window.open(passUrl, '_blank');
              if (!win) {
                const label = eventType === 'exit' ? 'Visitor exit pass' : 'Visitor pass';
                const ok = confirm(`${label} ready for ${pass.name || pass.id || 'visitor'}. Open now?`);
                if (ok) window.location.href = passUrl;
              }
              if (eventKey) {
                openedVisitorPasses.add(eventKey);
              }
              if (eventType === 'exit') {
                showToast(`Visitor exit pass ready: ${pass.id || pass.name || 'visitor'}`, 'good');
                addAudit(`Visitor exit detected: ${pass.id || pass.name || 'visitor'}`, 'info');
              } else {
                showToast(`Visitor pass issued: ${pass.id || pass.name || 'visitor'}`, 'good');
                addAudit(`Auto visitor pass issued: ${pass.id || pass.name || 'visitor'}`, 'info');
              }
            });
            if (maxSeen === dashboardState.lastAutoPassAt) {
              maxSeen = Date.now();
            }
            dashboardState.lastAutoPassAt = maxSeen;
          } catch (e) {
            // no-op
          }
        }

        function quickRemoveEmployee() {
          const key = (prompt('Employee ID or Name to remove:') || '').trim();
          if (!key) return;
          const payload = key.toUpperCase().startsWith('EMP')
            ? { emp_id: key }
            : { name: key };
          fetch('/api/employees/remove', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify(payload),
          })
            .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
            .then(({ ok, data }) => {
              if (!ok || !data.ok) {
                showToast(data.error || 'Failed to remove employee', 'bad');
                addAudit(`Remove employee failed: ${key}`, 'error');
                return;
              }
              const removed = (data.removed && data.removed.length) ? data.removed.join(', ') : key;
              showToast(`Employee removed: ${removed}`, 'warn');
              if (data.embeddings_updated) {
                showToast('Embeddings updated; retraining model...', 'good');
              }
              if (data.warning) {
                showToast(data.warning, 'warn');
              }
              if (data.retrain_error) {
                showToast(`Retrain failed: ${data.retrain_error}`, 'bad');
              }
              addAudit(`Employee removed: ${removed}`, 'warn');
            })
            .catch(() => {
              showToast('Failed to remove employee', 'bad');
              addAudit(`Remove employee failed: ${key}`, 'error');
            });
        }

        function quickIssueVisitorPass() {
          window.open('/visitors/new', '_blank');
        }

        function quickRegisterVisitorFace() {
          window.open('/visitors/register', '_blank');
        }

        function quickRebuildVisitorEmbeddings() {
          const name = (prompt('Visitor name to rebuild embeddings:') || '').trim();
          if (!name) return;
          fetch('/api/visitors/rebuild_embeddings', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({ name: name }),
          })
            .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
            .then(({ ok, data }) => {
              if (!ok || !data.ok) {
                showToast(data.error || 'Failed to rebuild embeddings', 'bad');
                addAudit(`Visitor embedding rebuild failed: ${name}`, 'error');
                return;
              }
              showToast(`Rebuild started for ${data.label || name}`, 'good');
              addAudit(`Visitor embedding rebuild started: ${data.label || name}`, 'info');
              refreshVisitorEmbeddingsStatus();
            })
            .catch(() => {
              showToast('Failed to rebuild embeddings', 'bad');
              addAudit(`Visitor embedding rebuild failed: ${name}`, 'error');
            });
        }

        function quickRetrainModel() {
          fetch('/api/model/retrain', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({}),
          })
            .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
            .then(({ ok, data }) => {
              if (!ok || !data.ok) {
                showToast(data.error || 'Failed to retrain model', 'bad');
                addAudit('Model retrain failed', 'error');
                return;
              }
              showToast('Model retraining started', 'good');
              addAudit('Model retraining started', 'info');
            })
            .catch(() => {
              showToast('Failed to retrain model', 'bad');
              addAudit('Model retrain failed', 'error');
            });
        }

        function quickRevokeVisitorPass() {
          const key = (prompt('Visitor pass ID or name to revoke:') || '').trim();
          if (!key) return;
          const payload = key.toUpperCase().startsWith('VIS')
            ? { id: key }
            : { name: key };
          fetch('/api/visitors/revoke', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify(payload),
          })
            .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
            .then(({ ok, data }) => {
              if (!ok || !data.ok) {
                showToast(data.error || 'Failed to revoke visitor pass', 'bad');
                addAudit(`Visitor pass revoke failed: ${key}`, 'error');
                return;
              }
              const removed = (data.removed && data.removed.length) ? data.removed.join(', ') : key;
              showToast(`Visitor pass revoked: ${removed}`, 'warn');
              addAudit(`Visitor pass revoked: ${removed}`, 'warn');
            })
            .catch(() => {
              showToast('Failed to revoke visitor pass', 'bad');
              addAudit(`Visitor pass revoke failed: ${key}`, 'error');
            });
        }

        function quickRemoveVisitorEmbedding() {
          const name = (prompt('Visitor name to remove from embeddings:') || '').trim();
          if (!name) return;
          fetch('/api/visitors/remove', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({ name: name }),
          })
            .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
            .then(({ ok, data }) => {
              if (!ok || !data.ok) {
                showToast(data.error || 'Failed to remove visitor from embeddings', 'bad');
                addAudit(`Visitor embedding removal failed: ${name}`, 'error');
                return;
              }
              showToast(`Visitor removed: ${data.removed || name}`, 'warn');
              if (data.retrain_error) {
                showToast(`Retrain failed: ${data.retrain_error}`, 'bad');
              }
              addAudit(`Visitor embeddings removed: ${data.removed || name}`, 'warn');
            })
            .catch(() => {
              showToast('Failed to remove visitor from embeddings', 'bad');
              addAudit(`Visitor embedding removal failed: ${name}`, 'error');
            });
        }

        function quickManualMark() {
          const name = (prompt('Employee name for manual mark:') || '').trim();
          if (!name) return;
          const now = new Date();
          dashboardState.manualMarks.push({
            name: name,
            date: dashboardState.selectedDate || todayIsoDate(),
            time: `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`,
          });
          showToast(`Manual mark noted for ${name}`, 'good');
          addAudit(`Manual mark created for ${name}`, 'manual');
          computeAnalytics();
        }

        function exportAttendanceCsv() {
          const rows = dashboardState.filteredAttendance || [];
          if (!rows.length) {
            showToast('No attendance rows to export', 'warn');
            return;
          }
          const header = ['Date', 'Name', 'Time'];
          const csvRows = [header.join(',')];
          rows.forEach((row) => {
            const values = [
              row.date || '',
              row.name || '',
              row.time || '',
            ].map((v) => `"${String(v).replace(/"/g, '""')}"`);
            csvRows.push(values.join(','));
          });
          const blob = new Blob([csvRows.join('\\n')], { type: 'text/csv;charset=utf-8' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `attendance_${dashboardState.selectedDate || todayIsoDate()}.csv`;
          document.body.appendChild(a);
          a.click();
          a.remove();
          URL.revokeObjectURL(url);
          showToast('Attendance CSV exported', 'good');
          addAudit('Exported attendance CSV', 'info');
        }

        function clearAttendanceFilters() {
          dashboardState.attendanceSearch = '';
          dashboardState.deptFilter = 'all';
          dashboardState.shiftFilter = 'all';
          dashboardState.statusFilter = 'all';
          dashboardState.selectedDate = todayIsoDate();
          const map = [
            ['attendanceSearch', dashboardState.attendanceSearch],
            ['filterDept', 'all'],
            ['filterShift', 'all'],
            ['filterStatus', 'all'],
            ['filterDate', dashboardState.selectedDate],
          ];
          map.forEach(([id, value]) => {
            const el = document.getElementById(id);
            if (el) el.value = value;
          });
          computeAnalytics();
          showToast('Attendance filters reset', 'good');
        }

        function toggleCameraConfig() {
          const panel = document.getElementById('cameraConfigPanel');
          panel.classList.toggle('hidden');
          if (!panel.classList.contains('hidden')) {
            const hostInput = document.getElementById('camHost');
            if (hostInput) hostInput.focus();
          }
        }

        async function addCamera(autoStart) {
          const payload = {
            camera_id: (document.getElementById('camId').value || '').trim(),
            name: (document.getElementById('camName').value || '').trim(),
            host: (document.getElementById('camHost').value || '').trim(),
            port: (document.getElementById('camPort').value || '').trim(),
            username: (document.getElementById('camUser').value || '').trim(),
            password: (document.getElementById('camPass').value || '').trim(),
            path: (document.getElementById('camPath').value || '').trim(),
            source: (document.getElementById('camSource').value || '').trim(),
            role: (document.getElementById('camRole').value || 'general').trim(),
            auto_start: !!autoStart,
          };

          setFormMessage('Submitting camera configuration...', 'warn');
          try {
            const res = await fetch('/api/cameras', {
              method: 'POST',
              headers: {'Content-Type':'application/json'},
              body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (!res.ok || !data.ok) {
              setFormMessage(data.error || 'Failed to add camera', 'bad');
              showToast(data.error || 'Failed to add camera', 'bad');
              return;
            }
            setFormMessage(`Camera added: ${data.camera?.camera_id || '-'}`, 'good');
            showToast(`Camera added: ${data.camera?.camera_id || '-'}`, 'good');
            addAudit(`Camera added: ${data.camera?.camera_id || '-'}`, 'info');
            await refreshAll();
          } catch (e) {
            setFormMessage('Failed to add camera', 'bad');
            showToast('Failed to add camera', 'bad');
            addAudit('Camera add failed', 'error');
          }
        }

        async function cameraAction(cameraId, action) {
          let url = '';
          let method = 'POST';
          if (action === 'start') url = `/api/cameras/${encodeURIComponent(cameraId)}/start`;
          if (action === 'stop') url = `/api/cameras/${encodeURIComponent(cameraId)}/stop`;
          if (action === 'remove') {
            if (!confirm(`Remove camera '${cameraId}'?`)) return;
            url = `/api/cameras/${encodeURIComponent(cameraId)}`;
            method = 'DELETE';
          }

          try {
            const res = await fetch(url, { method });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            if (action === 'start') showToast(`Started ${cameraId}`, 'good');
            if (action === 'stop') showToast(`Stopped ${cameraId}`, 'warn');
            if (action === 'remove') showToast(`Removed ${cameraId}`, 'warn');
            addAudit(`Camera action: ${action} (${cameraId})`, 'info');
            await refreshAll();
          } catch (e) {
            showToast(`Failed action: ${action}`, 'bad');
            addAudit(`Camera action failed: ${action} (${cameraId})`, 'error');
          }
        }

        async function stopAllCameras() {
          try {
            const res = await fetch('/api/stop_all', { method: 'POST' });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            showToast('Stop signal sent to all cameras', 'warn');
            addAudit('Issued stop-all cameras command', 'warn');
            await refreshAll();
          } catch (e) {
            showToast('Failed to stop all cameras', 'bad');
            addAudit('Failed stop-all cameras command', 'error');
          }
        }

        function timeToSeconds(raw) {
          const parts = String(raw || '').split(':').map((v) => Number(v));
          if (parts.length < 2 || parts.some((v) => Number.isNaN(v))) return Number.MAX_SAFE_INTEGER;
          const h = parts[0] || 0;
          const m = parts[1] || 0;
          const s = parts[2] || 0;
          return h * 3600 + m * 60 + s;
        }

        function rebuildEmployeeRegistry() {
          const uniqueNames = [];
          const seen = new Set();
          (dashboardState.attendance || []).forEach((r) => {
            const name = String(r['emp name'] || '').trim();
            if (!name) return;
            const key = name.toLowerCase();
            if (seen.has(key)) return;
            seen.add(key);
            uniqueNames.push(name);
          });

          (dashboardState.manualMarks || []).forEach((m) => {
            const name = String(m.name || '').trim();
            if (!name) return;
            const key = name.toLowerCase();
            if (seen.has(key)) return;
            seen.add(key);
            uniqueNames.push(name);
          });

          dashboardState.roster = uniqueNames.sort((a, b) => a.localeCompare(b));
        }

        function calculateStreak(name) {
          if (!name) return 0;
          const dates = new Set();
          (dashboardState.attendance || []).forEach((r) => {
            if (String(r['emp name'] || '').trim().toLowerCase() === name.toLowerCase()) {
              if (r.date) dates.add(String(r.date));
            }
          });
          const sorted = Array.from(dates).sort().reverse();
          if (!sorted.length) return 0;
          let streak = 0;
          let cursor = new Date(`${dashboardState.selectedDate || todayIsoDate()}T00:00:00`);
          for (let i = 0; i < 365; i++) {
            const iso = `${cursor.getFullYear()}-${String(cursor.getMonth() + 1).padStart(2, '0')}-${String(cursor.getDate()).padStart(2, '0')}`;
            if (dates.has(iso)) streak += 1;
            else break;
            cursor.setDate(cursor.getDate() - 1);
          }
          return streak;
        }

        function applyAttendanceFilters() {
          const selectedDate = dashboardState.selectedDate || todayIsoDate();
          const dateRows = (dashboardState.attendance || []).filter((r) => String(r.date || '') === selectedDate);
          const baseRows = dateRows.map((row) => {
            const name = String(row['emp name'] || row.name || '').trim();
            return {
              date: row.date || '',
              name: name,
              time: row.time || '',
            };
          });

          dashboardState.manualMarks
            .filter((m) => m.date === selectedDate)
            .forEach((m) => {
              const name = String(m.name || '').trim();
              baseRows.push({
                date: selectedDate,
                name: name,
                time: m.time,
              });
            });

          dashboardState.filteredAttendance = baseRows.filter((r) => {
            if (dashboardState.attendanceSearch && !String(r.name || '').toLowerCase().includes(dashboardState.attendanceSearch)) return false;
            return true;
          });
        }

        function renderCameraHealth() {
          const host = document.getElementById('cameraHealthList');
          if (!host) return;
          const cams = dashboardState.cameras || [];
          if (!cams.length) {
            host.innerHTML = '<div class="list-item"><p class="list-title">No camera health data yet</p><p class="list-meta">Start a camera to populate health metrics.</p></div>';
            return;
          }
          host.innerHTML = '';
          cams.forEach((cam) => {
            const item = document.createElement('div');
            item.className = 'list-item';
            const stateChip = cam.running ? '<span class="status-chip on-time">Online</span>' : '<span class="status-chip absent">Offline</span>';
            item.innerHTML = `
              <p class="list-title">${escapeHtml(cam.name || cam.camera_id)} ${stateChip}</p>
              <p class="list-meta">FPS ${Number(cam.fps || 0).toFixed(2)} | Queue ${cam.queue_size ?? 0} | Tracks ${cam.active_tracks ?? 0} | Last frame ${escapeHtml(cam.last_frame_at || '-')}</p>
            `;
            host.appendChild(item);
          });
        }

        function renderExceptions() {
          const host = document.getElementById('exceptionsList');
          if (!host) return;
          const issues = [];

          const offline = (dashboardState.cameras || []).filter((c) => !c.running).length;
          if (offline > 0) issues.push({ title: `${offline} camera(s) offline`, meta: 'Check camera health and network link.' });

          const highQueue = (dashboardState.cameras || []).filter((c) => Number(c.queue_size || 0) > 2).length;
          if (highQueue > 0) issues.push({ title: `${highQueue} camera(s) with queue delay`, meta: 'Recognition queue is above recommended level.' });

          const unknownEstimate = (dashboardState.cameras || []).reduce((acc, c) => {
            return acc + Math.max(0, Number(c.active_tracks || 0) - Number(c.known_tracks || 0));
          }, 0);
          if (unknownEstimate > 0) issues.push({ title: `${unknownEstimate} unknown track(s) estimated`, meta: 'Unknown faces detected in active streams.' });

          let duplicatePunches = 0;
          const byName = {};
          (dashboardState.filteredAttendance || []).forEach((r) => {
            const key = r.name.toLowerCase();
            if (!byName[key]) byName[key] = [];
            byName[key].push(timeToSeconds(r.time));
          });
          Object.values(byName).forEach((times) => {
            times.sort((a, b) => a - b);
            for (let i = 1; i < times.length; i++) {
              if (times[i] - times[i - 1] <= 60) duplicatePunches += 1;
            }
          });
          if (duplicatePunches > 0) issues.push({ title: `${duplicatePunches} duplicate punch event(s)`, meta: 'Multiple marks within 60 seconds.' });

          if (!issues.length) {
            host.innerHTML = '<div class="list-item"><p class="list-title">No exceptions</p><p class="list-meta">System is operating normally.</p></div>';
            return;
          }

          host.innerHTML = '';
          issues.forEach((it) => {
            const el = document.createElement('div');
            el.className = 'list-item';
            el.innerHTML = `<p class="list-title">${escapeHtml(it.title)}</p><p class="list-meta">${escapeHtml(it.meta)}</p>`;
            host.appendChild(el);
          });
        }

        function renderTimeline() {
          const host = document.getElementById('timelineGrid');
          if (!host) return;
          const buckets = new Array(12).fill(0); // 8:00 to 19:59
          (dashboardState.filteredAttendance || []).forEach((r) => {
            const h = Number(String(r.time || '0').split(':')[0]);
            const idx = h - 8;
            if (idx >= 0 && idx < buckets.length) buckets[idx] += 1;
          });
          const maxVal = Math.max(1, ...buckets);
          host.innerHTML = '';
          buckets.forEach((val, idx) => {
            const col = document.createElement('div');
            col.className = 'hour-col';
            const hh = String(idx + 8).padStart(2, '0');
            const pct = Math.max(4, Math.round((val / maxVal) * 100));
            col.innerHTML = `
              <div class="hour-bar-shell"><div class="hour-bar" style="height:${pct}%"></div></div>
              <div class="hour-label">${hh}:00</div>
              <div class="hour-val">${val}</div>
            `;
            host.appendChild(col);
          });
        }

        function renderAuditTrail() {
          const host = document.getElementById('auditList');
          if (!host) return;
          if (!dashboardState.auditTrail.length) {
            host.innerHTML = '<div class="list-item"><p class="list-title">Audit log initialized</p><p class="list-meta">Waiting for actions...</p></div>';
            return;
          }
          host.innerHTML = '';
          dashboardState.auditTrail.forEach((event) => {
            const el = document.createElement('div');
            el.className = 'list-item';
            const at = new Date(event.at);
            const stamp = `${at.toLocaleDateString()} ${at.toLocaleTimeString()}`;
            el.innerHTML = `<p class="list-title">${escapeHtml(event.message)}</p><p class="list-meta">${escapeHtml(stamp)} | ${escapeHtml(event.level || 'info')}</p>`;
            host.appendChild(el);
          });
        }

        function computeAnalytics() {
          rebuildEmployeeRegistry();
          applyAttendanceFilters();

          const selected = dashboardState.selectedDate || todayIsoDate();
          const selectedRows = (dashboardState.attendance || []).filter((r) => String(r.date || '') === selected);
          const presentSet = new Set();
          selectedRows.forEach((r) => {
            const name = String(r['emp name'] || r.name || '').trim().toLowerCase();
            if (name) presentSet.add(name);
          });
          dashboardState.manualMarks
            .filter((m) => m.date === selected)
            .forEach((m) => {
              const name = String(m.name || '').trim().toLowerCase();
              if (name) presentSet.add(name);
            });
          const rosterCount = dashboardState.roster.length;
          const presentCount = presentSet.size;
          const absentCount = Math.max(0, rosterCount - presentCount);
          document.getElementById('totalEmployeesPill').textContent = String(rosterCount);
          document.getElementById('presentPill').textContent = String(presentCount);
          document.getElementById('absentPill').textContent = String(absentCount);

          const totalSub = document.getElementById('totalEmployeesSub');
          if (totalSub) totalSub.textContent = `Roster records: ${rosterCount}`;
          const pSub = document.getElementById('presentSub');
          if (pSub) pSub.textContent = `Date: ${selected}`;
          const aSub = document.getElementById('absentSub');
          if (aSub) aSub.textContent = `Date: ${selected}`;
        }

        function upsertCameraCard(cam) {
          const grid = document.getElementById('cameraGrid');
          const id = cameraCardId(cam.camera_id);
          let card = document.getElementById(id);

          if (!card) {
            card = document.createElement('div');
            card.className = 'cam-card';
            card.id = id;
            card.innerHTML = `
              <div class="cam-head">
                <div class="cam-meta">
                  <div class="cam-title" data-k="title"></div>
                  <div class="cam-sub" data-k="sub"></div>
                </div>
                <span class="run-badge" data-k="run"></span>
              </div>
              <div class="cam-media">
                <img class="cam-img" data-k="img" alt="camera stream" />
                <div class="cam-tools">
                  <button class="subtle" data-k="startBtn">Start</button>
                  <button class="subtle" data-k="stopBtn">Stop</button>
                  <button class="danger" data-k="removeBtn">Remove</button>
                </div>
              </div>
              <div class="cam-foot">
                <span class="mini" data-k="fps"></span>
                <span class="mini" data-k="tracks"></span>
                <span class="mini" data-k="known"></span>
                <span class="mini" data-k="queue"></span>
                <span class="mini" data-k="err"></span>
              </div>
            `;
            grid.appendChild(card);
            card.querySelector('[data-k="startBtn"]').onclick = () => cameraAction(cam.camera_id, 'start');
            card.querySelector('[data-k="stopBtn"]').onclick = () => cameraAction(cam.camera_id, 'stop');
            card.querySelector('[data-k="removeBtn"]').onclick = () => cameraAction(cam.camera_id, 'remove');
          }

          card.querySelector('[data-k="title"]').textContent = `${cam.name || cam.camera_id} (${cam.camera_id})`;
          const roleLabel = cam.role ? ` | ${cam.role}` : '';
          card.querySelector('[data-k="sub"]').textContent = `${cam.source || '-'}${roleLabel}`;

          const runEl = card.querySelector('[data-k="run"]');
          runEl.textContent = cam.running ? 'RUNNING' : 'STOPPED';
          runEl.className = `run-badge ${cam.running ? 'on' : 'off'}`;

          card.querySelector('[data-k="fps"]').textContent = `FPS: ${Number(cam.fps || 0).toFixed(2)}`;
          card.querySelector('[data-k="tracks"]').textContent = `Tracks: ${cam.active_tracks ?? 0}`;
          card.querySelector('[data-k="known"]').textContent = `Known: ${cam.known_tracks ?? 0}`;
          const queueEl = card.querySelector('[data-k="queue"]');
          queueEl.textContent = `Queue: ${cam.queue_size ?? 0}`;
          queueEl.className = `mini ${Number(cam.queue_size || 0) > 2 ? 'warn' : ''}`.trim();

          const errEl = card.querySelector('[data-k="err"]');
          if (cam.error) {
            errEl.textContent = `Error: ${cam.error}`;
            errEl.className = 'mini error';
          } else {
            errEl.textContent = 'Error: none';
            errEl.className = 'mini';
          }

          const img = card.querySelector('[data-k="img"]');
          const expectedPrefix = `${location.origin}/video_feed/${encodeURIComponent(cam.camera_id)}`;
          if (cam.running) {
            if (!img.src || !img.src.startsWith(expectedPrefix)) {
              img.src = `/video_feed/${encodeURIComponent(cam.camera_id)}?ts=${Date.now()}`;
            }
          } else {
            img.removeAttribute('src');
          }
        }

        function removeMissingCards(currentIds) {
          const set = new Set(currentIds.map((v) => cameraCardId(v)));
          const cards = Array.from(document.querySelectorAll('.cam-card'));
          cards.forEach((el) => {
            if (!set.has(el.id)) el.remove();
          });
        }

        function getFilteredCameras() {
          return dashboardState.cameras.filter((cam) => {
            const haystack = `${cam.name || ''} ${cam.camera_id || ''} ${cam.source || ''}`.toLowerCase();
            if (dashboardState.cameraSearch && !haystack.includes(dashboardState.cameraSearch)) return false;
            if (dashboardState.cameraFilter === 'running' && !cam.running) return false;
            if (dashboardState.cameraFilter === 'stopped' && cam.running) return false;
            return true;
          });
        }

        function renderCameras() {
          const filtered = getFilteredCameras();
          filtered.forEach(upsertCameraCard);
          removeMissingCards(filtered.map((c) => c.camera_id));
          const empty = document.getElementById('cameraEmpty');
          if (empty) empty.style.display = filtered.length ? 'none' : 'block';
        }

        async function refreshCameras() {
          const res = await fetch('/api/cameras');
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const cams = await res.json();
          dashboardState.cameras = Array.isArray(cams) ? cams : [];
          renderCameras();
        }

        async function refreshGlobalStatus() {
          const res = await fetch('/api/status');
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();
          const activeSub = document.getElementById('activeSub');
          if (activeSub) {
            activeSub.textContent = `Running ${data.running_cameras ?? 0} / ${data.total_cameras ?? 0}`;
          }
        }

        function renderAttendance() {
          const tbody = document.getElementById('rowsBody');
          if (!tbody) return;
          tbody.innerHTML = '';
          const rows = (dashboardState.filteredAttendance || []).slice().sort((a, b) => {
            return timeToSeconds(a.time) - timeToSeconds(b.time);
          });
          if (!rows.length) {
            tbody.innerHTML = '<tr><td colspan="3" class="muted">No attendance rows for current filter</td></tr>';
            return;
          }
          rows.forEach((r) => {
            const tr = document.createElement('tr');
            tr.innerHTML =
              `<td>${escapeHtml(r.date || '-')}</td>` +
              `<td>${escapeHtml(r.name || '-')}</td>` +
              `<td>${escapeHtml(r.time || '-')}</td>`;
            tbody.appendChild(tr);
          });
        }

        async function refreshAttendance() {
          const res = await fetch(`/api/attendance?limit=${encodeURIComponent(dashboardState.attendanceLimit)}`);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const rows = await res.json();
          dashboardState.attendance = Array.isArray(rows) ? rows : [];
        }

        async function refreshAll(manual) {
          if (refreshInFlight) return;
          refreshInFlight = true;
          try {
            await Promise.all([refreshCameras(), refreshAttendance(), refreshVisitorEmbeddingsStatus()]);
            computeAnalytics();
            dashboardState.lastSyncAt = Date.now();
            updateSyncStaleness();
            checkAutoVisitorPasses();
            if (manual) {
              showToast('Dashboard refreshed', 'good');
              addAudit('Dashboard refreshed manually', 'info');
            }
          } catch (e) {
            setSyncBadge('bad', 'Snapshot fetch failed');
            if (manual) {
              showToast('Failed to refresh dashboard', 'bad');
              addAudit('Dashboard refresh failed', 'error');
            }
          } finally {
            refreshInFlight = false;
          }
        }

        function bindControls() {
          const cameraSearch = document.getElementById('cameraSearch');
          const cameraFilter = document.getElementById('cameraFilter');
          const attendanceSearch = document.getElementById('attendanceSearch');
          const attendanceLimit = document.getElementById('attendanceLimit');
          const filterDate = document.getElementById('filterDate');
          const filterDept = document.getElementById('filterDept');
          const filterShift = document.getElementById('filterShift');
          const filterStatus = document.getElementById('filterStatus');

          dashboardState.selectedDate = todayIsoDate();
          if (filterDate) filterDate.value = dashboardState.selectedDate;

          if (cameraSearch) {
            cameraSearch.addEventListener('input', (ev) => {
              dashboardState.cameraSearch = String(ev.target.value || '').trim().toLowerCase();
              renderCameras();
            });
          }
          if (cameraFilter) {
            cameraFilter.addEventListener('change', (ev) => {
              dashboardState.cameraFilter = String(ev.target.value || 'all');
              renderCameras();
            });
          }
          if (attendanceSearch) {
            attendanceSearch.addEventListener('input', (ev) => {
              dashboardState.attendanceSearch = String(ev.target.value || '').trim().toLowerCase();
              computeAnalytics();
            });
          }
          if (filterDate) {
            filterDate.addEventListener('change', (ev) => {
              dashboardState.selectedDate = String(ev.target.value || todayIsoDate());
              computeAnalytics();
            });
          }
          if (filterDept) {
            filterDept.addEventListener('change', (ev) => {
              dashboardState.deptFilter = String(ev.target.value || 'all');
              computeAnalytics();
            });
          }
          if (filterShift) {
            filterShift.addEventListener('change', (ev) => {
              dashboardState.shiftFilter = String(ev.target.value || 'all');
              computeAnalytics();
            });
          }
          if (filterStatus) {
            filterStatus.addEventListener('change', (ev) => {
              dashboardState.statusFilter = String(ev.target.value || 'all');
              computeAnalytics();
            });
          }
          if (attendanceLimit) {
            dashboardState.attendanceLimit = Number(attendanceLimit.value || 100);
            attendanceLimit.addEventListener('change', (ev) => {
              dashboardState.attendanceLimit = Math.max(1, Number(ev.target.value || 100));
              refreshAll(true);
            });
          }
        }

        bindControls();
        addAudit('UI initialized', 'info');
        refreshAll();
        setInterval(refreshAll, 2500);
        setInterval(updateSyncStaleness, 1000);

        async function refreshVisitorEmbeddingsStatus() {
          const statusEl = document.getElementById('visitorEmbedStatus');
          if (!statusEl) return;
          try {
            const res = await fetch('/api/visitors/embeddings/status');
            if (!res.ok) return;
            const data = await res.json();
            const labels = Array.isArray(data.labels) ? data.labels : [];
            const running = !!data.running;
            const tail = Array.isArray(data.log_tail) ? data.log_tail : [];
            const lastLine = tail.length ? tail[tail.length - 1] : '';
            const labelText = labels.length ? ` (${labels.join(', ')})` : '';
            if (running) {
              statusEl.className = 'sync-status warn';
              statusEl.textContent = `Visitor embeddings: running${labelText}${lastLine ? ` | ${lastLine}` : ''}`;
            } else {
              statusEl.className = 'sync-status';
              statusEl.textContent = `Visitor embeddings: idle${lastLine ? ` | ${lastLine}` : ''}`;
            }
          } catch (e) {
            // no-op
          }
        }
