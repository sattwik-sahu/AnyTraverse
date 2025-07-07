from anytraverse.utils.cli.human_op.hoc_ctx import AnyTraverseHOC_Context, DriveStatus
from websockets.sync.server import serve, ServerConnection

from threading import Lock
from typing import Callable
import json


class AnyTraverseWebsocket:
    _anytraverse: AnyTraverseHOC_Context
    _human_call: Callable[[DriveStatus], None]

    _hostname: str
    _port: int
    _clients: list[ServerConnection]

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

    def _broadcast(self, msg: str) -> None:
        for client in self._clients:
            client.send(msg)

    def human_call(self, status: DriveStatus) -> None:
        msg = json.dumps({"msg": "HOC required!", "type": status.value})
        self._broadcast(msg=msg)

    def _handler(self, websocket: ServerConnection) -> None:
        for message in websocket:
            print(websocket)
            print(f"Message >>> {message} [type: {type(message)}]")
            try:
                self._lock.acquire()
                self._anytraverse.human_call_with_syntax(prompts_str=str(message))  # type: ignore
            except Exception as ex:
                print("Error in human operator call")
                raise ex
            finally:
                self._lock.release()

            self._broadcast(
                msg=f"Updated prompts: {str(dict(self._anytraverse.prompts))}"
            )

    def start(self) -> None:
        with serve(
            handler=self._handler, host=self._hostname, port=self._port
        ) as self._server:
            print(
                f"Starting AnyTraverse HOC server on ws://{self._hostname}:{self._port}"
            )

            self._server.serve_forever()

    def shutdown(self) -> None:
        self._server.shutdown()


# if __name__ == "__main__":
#     anytraverse = create_anytraverse_hoc_context(init_prompts=[("grass", 1.0)])
#     print("Creating AnyTraverse...")
#     ws_hoc = AnyTraverseWebsocket(anytraverse=anytraverse, port=7777)
#     print("Created AnyTraverse human operator pipeline")

#     print("Starting server...")
#     thread = Thread(target=ws_hoc.start)
#     thread.start()
#     print("Done...")

#     print("Listening for changes to prompts...")
#     ws_hoc.lock.acquire()
#     n_prompts: int = len(anytraverse.prompts)
#     ws_hoc.lock.release()
#     while True:
#         ws_hoc.lock.acquire()
#         if len(anytraverse.prompts) > n_prompts:
#             print(f"Updated prompts >>> {anytraverse.prompts}")
#             ws_hoc.lock.release()
#             break
#         ws_hoc.lock.release()

#     ws_hoc.shutdown()
#     print(f"Final prompts >>> {anytraverse.prompts}")
