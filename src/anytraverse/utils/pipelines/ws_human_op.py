from anytraverse.utils.cli.human_op.hoc_ctx import AnyTraverseHOC_Context
from websockets.sync.server import serve, ServerConnection
from threading import Lock


class AnyTraverseWebsocket:
    _anytraverse: AnyTraverseHOC_Context

    _hostname: str
    _port: int

    _lock: Lock

    def __init__(
        self,
        anytraverse: AnyTraverseHOC_Context,
        port: int = 6969,
        hostname: str = "0.0.0.0",
    ) -> None:
        self._hostname = hostname
        self._port = port
        self._clients = []
        self._anytraverse = anytraverse
        self._lock = Lock()

    @property
    def lock(self) -> Lock:
        return self._lock

    def human_op_call(self, websocket: ServerConnection) -> None:
        self._lock.acquire()
        try:
            for message in websocket:
                self._anytraverse.human_call_with_syntax(prompts_str=message)  # type: ignore
                websocket.send(message=str(dict(self._anytraverse.prompts)))
        except Exception:
            print("Error in human operator call")
        finally:
            self._lock.release()

    def start(self) -> None:
        with serve(
            handler=self.human_op_call, host=self._hostname, port=self._port
        ) as server:
            print(
                f"Starting AnyTraverse HOC server on ws://{self._hostname}:{self._port}"
            )
            server.serve_forever()
