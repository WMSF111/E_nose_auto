class Event(object):
    '''
      事件初始化的一个方式
    '''

    def __init__(self, event_type, data=None):
        self._type = event_type
        self._data = data

    @property
    def type(self):
        return self._type

    @property
    def data(self):
        return self._data


class EventDispatcher(object):
    """
   event分发类 监听和分发event事件
   """

    def __init__(self):
        # 初始化事件
        self._events = dict()

    def __del__(self):
        self._events = None

    def has_listener(self, event_type, listener):
        if event_type in self._events.keys():
            return listener in self._events[event_type]
        else:
            return False

    def dispatch_event(self, event):
        """
      Dispatch an instance of Event class
      """
        # 分发event到所有关联的listener
        if event.type in self._events.keys():
            listeners = self._events[event.type]

            for listener in listeners:
                listener(event)

    def add_event_listener(self, event_type, listener):
        # 给某种事件类型添加listner
        if not self.has_listener(event_type, listener):
            listeners = self._events.get(event_type, [])
            listeners.append(listener)
            self._events[event_type] = listeners

    def remove_event_listener(self, event_type, listener):
        if self.has_listener(event_type, listener):
            listeners = self._events[event_type]
            if len(listeners) == 1:
                del self._events[event_type]
            else:
                listeners.remove(listener)
                self._events[event_type] = listeners


class PKGEvent(Event):
    PKG_DATA_OK = "PKG_DATA_OK"


if __name__ == '__main__':
    class Test(EventDispatcher):
        def __init__(self, a=1, b=1):
            EventDispatcher.__init__(self)
            self.a = a
            self.b = b

        def testDispEvent(self):
            self.dispatch_event(PKGEvent(PKGEvent.PKG_DATA_OK, 124))


    t = Test()


    def on_ok(e):
        print(e.data)


    t.add_event_listener(PKGEvent.PKG_DATA_OK, on_ok)
    import time

    while True:
        print(1)
        t.testDispEvent()
        time.sleep(1)