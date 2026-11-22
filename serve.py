"""Serve this folder on http://localhost:8000 with the headers that enable multi-threaded WebAssembly.

    python3 serve.py [port]

Byte-range requests are supported, so the video also plays in Safari.
"""
import http.server, os, re, sys

class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, '.wasm': 'application/wasm',
                      '.mjs': 'text/javascript', '.js': 'text/javascript', '.onnx': 'application/octet-stream',
                      '.mp4': 'video/mp4'}
    _left = None

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def send_head(self):
        self._left = None
        rng, path = self.headers.get('Range'), self.translate_path(self.path)
        m = re.fullmatch(r'bytes=(\d*)-(\d*)', (rng or '').strip())
        if not m or not os.path.isfile(path) or m.groups() == ('', ''):
            return super().send_head()
        size = os.path.getsize(path)
        a, b = m.groups()
        start, end = (max(0, size - int(b)), size - 1) if a == '' else (int(a), min(int(b), size - 1) if b else size - 1)
        if start >= size or start > end:
            self.send_response(416); self.send_header('Content-Range', f'bytes */{size}'); self.end_headers()
            return None
        f = open(path, 'rb'); f.seek(start)
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end - start + 1))
        self.end_headers()
        self._left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        if self._left is None:
            return super().copyfile(source, outputfile)
        while self._left > 0:
            buf = source.read(min(1 << 16, self._left))
            if not buf:
                break
            outputfile.write(buf); self._left -= len(buf)

    def log_message(self, *a):
        pass

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
print(f'open http://localhost:{port}/')
http.server.ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
