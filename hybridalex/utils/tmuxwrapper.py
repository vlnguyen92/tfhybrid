from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import subprocess as sp


@contextlib.contextmanager
def tmux(session, cwd=None, destroy=False):
    """run commands in tmux session"""
    def remote_run(*args, **kwargs):
        """wrapper around fabric run"""
        if 'quiet' in kwargs:
            del kwargs['quiet']
        kwargs['shell'] = True
        kwargs['universal_newlines'] = True
        kwargs['stdout'] = sp.PIPE
        return sp.run(*args, **kwargs)

    class tmux_session(object):
        """tmux session object"""

        def __init__(self, session, cwd):
            super(tmux_session, self).__init__()
            # tmux_cmd = 'tmux -CC'
            tmux_cmd = 'tmux'
            self.session = session
            self.cwd = cwd if cwd is not None else ''
            self.environ = {}
            patterns = {
                'cmd': ['send-keys', '-t {se}{{wd}}{{pane}}',
                        "'cd {{cwd}}' C-m",
                        'env Space {{envvar}} Space',
                        "'{{cmd}}' C-m",
                        ],
                'cmdnoenv': ['send-keys', '-t {se}{{wd}}{{pane}}',
                             "'cd {{cwd}}' C-m",
                             "'{{cmd}}' C-m",
                             ],
                'neww': ['new-window', '-t {se} -n {{name}}'],
                'has': ['has-session', '-t {se}'],
                'new': ['new-session', '-d', '-s {se}'],
                'kill': ['kill-session', '-t {se}'],
                'killw': ['kill-window', '-t {se}:{{name}}'],
                'lsw': ['list-windows', '-F \'#W\'', '-t {se}'],
                'selectw': ['select-window', '-t {se}:{{name}}'],
            }
            self.patterns = {k: ' '.join([tmux_cmd] + v).format(se=session)
                             for k, v in patterns.items()}
            self._ensure_session()

        def _has_session(self):
            """has session"""
            return remote_run(self.patterns['has'], quiet=True).returncode == 0

        def _ensure_session(self):
            """Ensure session exists"""
            if not self._has_session():
                return remote_run(self.patterns['new'])
            return True

        def _kill_session(self):
            """Kill session"""
            if self._has_session():
                return remote_run(self.patterns['kill'])
            return True

        def _kill_window(self, name):
            """Kill window"""
            return remote_run(self.patterns['killw'].format(name=name), quiet=True)

        def _run_in_pane(self, cmd, window=None, pane=None, noenv=False):
            """Run commands in tmux window"""
            window = ':' + window if window is not None else ''
            pane = '.' + pane if window != '' and pane is not None else ''
            envvar = ' Space '.join(["{}= \"'\"'{}'\"'\"".format(k, v)
                                     for k, v in self.environ.items()])
            cwd = self.cwd

            cwd = cwd.replace("'", r"\'")
            cmd = cmd.replace("'", "'\"'\"'")
            return remote_run(self.patterns['cmdnoenv' if noenv else 'cmd']
                              .format(wd=window, pane=pane, envvar=envvar, cwd=cwd, cmd=cmd))

        def _new_window(self, name):
            """Create a new window"""
            return remote_run(self.patterns['neww'].format(name=name))

        def _list_windows(self):
            """List windows names in session"""
            return remote_run(self.patterns['lsw']).stdout.split()

        def _select_window(self, name):
            """Select window as current window"""
            return remote_run(self.patterns['selectw'].format(name=name))

        def run(self, cmd, new_window=None, noenv=False):
            """Run commands in tmux"""
            if new_window is not None:
                if new_window not in self._list_windows():
                    self._new_window(new_window)
                self._select_window(new_window)
            return self._run_in_pane(cmd, window=new_window, noenv=noenv)

        def kill(self, window=None):
            """Kill something"""
            if window is not None:
                return self._kill_window(window)

        def destroy(self):
            """Kill session"""
            return self._kill_session()

        @contextlib.contextmanager
        def cd(self, path):
            """Context manager for cd"""
            last_cwd = self.cwd
            if os.path.isabs(path):
                self.cwd = path
            else:
                self.cwd = os.path.normpath(os.path.join(self.cwd, path))
            try:
                yield
            finally:
                self.cwd = last_cwd

        @contextlib.contextmanager
        def env(self, clean_revert=False, **kwargs):
            """Set environment variables"""
            previous = {}
            new = []
            for key, value in kwargs.items():
                if key in self.environ:
                    previous[key] = self.environ[key]
                else:
                    new.append(key)
                self.environ[key] = value
            try:
                yield
            finally:
                if clean_revert:
                    for key, value in kwargs.items():
                        # If the current env value for this key still matches the
                        # value we set it to beforehand, we are OK to revert it to the
                        # pre-block value.
                        if key in self.environ and value == self.environ[key]:
                            if key in previous:
                                self.environ[key] = previous[key]
                            else:
                                del self.environ[key]
                else:
                    self.environ.update(previous)
                    for key in new:
                        del self.environ[key]

    ts = tmux_session(session, cwd)
    try:
        yield ts
    finally:
        if destroy:
            ts.destroy()
