
import pandas as pd
from typing import Any, Dict

from market_monitor.publishers.base import MessageType


class QueueDataSource:
    """Classe base per gestire code con diversi tipi di messaggi"""


    def __init__(self, queue):
        self._queue = queue
        self._handlers = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Registra i gestori predefiniti"""
        for msg_type, handler_name in MESSAGE_TYPES.items():
            if hasattr(self, handler_name):
                self._handlers[msg_type] = getattr(self, handler_name)

    def register_handler(self, message_type: MessageType, handler_func):
        """Permette di registrare handler personalizzati"""
        if isinstance(message_type, str):
            try:
                message_type = MessageType[message_type]
            except KeyError:
                raise KeyError(f'Message type "{message_type}" non trovato')
        self._handlers[message_type] = handler_func

    def get_data(self) -> pd.DataFrame:
        """Retrieve the DataFrame from a queue."""
        while not self._queue.empty():
            item = self._queue.get()
            self._process_message(item)

            # Se c'è un DataFrame, lo restituisce
            if 'data' in item and item['data'] is not None:
                return item['data']

        return pd.DataFrame()

    def _process_message(self, item: Dict[str, Any]):
        """Processa un messaggio in base al suo tipo"""
        msg_type = item.get('type', 'data')

        if msg_type in self._handlers:
            self._handlers[msg_type](item)
        else:
            self.handle_unknown(item)

    # Handler predefiniti
    def handle_data(self, item: Dict[str, Any]):
        """Gestisce messaggi di tipo 'data'"""
        pass  # Il DataFrame viene già gestito in get_dataframe

    def handle_command(self, item: Dict[str, Any]):
        """Gestisce comandi (es: pause, resume, stop)"""
        command = item.get('command')
        print(f"Comando ricevuto: {command}")

    def handle_config(self, item: Dict[str, Any]):
        """Gestisce configurazioni"""
        config = item.get('config')
        print(f"Configurazione ricevuta: {config}")

    def handle_status(self, item: Dict[str, Any]):
        """Gestisce messaggi di stato"""
        status = item.get('status')
        print(f"Status ricevuto: {status}")

    def handle_error(self, item: Dict[str, Any]):
        """Gestisce errori"""
        error = item.get('error')
        print(f"Errore ricevuto: {error}")

    def handle_unknown(self, item: Dict[str, Any]):
        """Gestisce messaggi di tipo sconosciuto"""
        print(f"Tipo di messaggio sconosciuto: {item}")

    def get_queue(self):
        return self._queue
