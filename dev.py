import asyncio
import http.server
import socketserver
import websockets
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Timer


# Disable caching in HTTP server
class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header(
            "Cache-Control", "no-store, no-cache, must-revalidate, max-age=0"
        )
        super().end_headers()


def run_http_server():
    PORT = 8000
    Handler = NoCacheHTTPRequestHandler
    httpd = socketserver.TCPServer(("", PORT), Handler)
    print(f"Serving HTTP on port {PORT}")
    httpd.serve_forever()


clients = set()


async def websocket_handler(websocket, path):
    clients.add(websocket)
    try:
        async for message in websocket:
            pass
    finally:
        clients.remove(websocket)


async def notify_clients(message):
    print(f"notify {len(clients)} clients", message)
    if clients:  # Send the message to all connected clients
        for client in clients:
            await client.send(message)


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, loop):
        self.loop = loop
        self.debounce_timer = None
        self.last_event = None

    def _debounced_notify(self):
        if self.last_event:
            asyncio.run_coroutine_threadsafe(notify_clients("update"), self.loop)

    def on_any_event(self, event):
        if event.event_type != "closed":
            return
        print(event)
        self.last_event = event
        if self.debounce_timer:
            self.debounce_timer.cancel()

        self.debounce_timer = Timer(0.1, self._debounced_notify)
        self.debounce_timer.start()


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    websocket_server = websockets.serve(websocket_handler, "localhost", 6789)
    loop.run_until_complete(websocket_server)

    from threading import Thread

    http_thread = Thread(target=run_http_server)
    http_thread.start()

    event_handler = FileChangeHandler(loop)
    observer = Observer()
    observer.schedule(event_handler, path=".", recursive=False)
    observer.start()

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
