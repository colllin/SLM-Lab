import numpy as np
import collections

class Eventful(object):
    """Decorator for classes which can emit events which can be "subscribed" to.
    
    Usage:
        @Eventful('STARTED', 'FINISHED', 'ERROR')
        class SomeClass:
            def someMethod(self):
                self.emit('FINISHED', {'data': 123})

        someInstance = SomeClass()
 
        def someCallback(data):
            print(data)
        someInstance.subscribe('FINISHED', someCallback)
    """

    def __init__(self, *event_names):
        """
        event_names: A list of str event names which the decorated class can emit.
        """
        assert len(event_names) > 0
        assert np.all([isintance(name, str) for name in event_names])
        self.event_names = event_names

    def __call__(self, cls):
        class EventfulWrap(cls):
            _events = self.event_names
            _event_callbacks = collections.defaultdict(list)
            def subscribe(self, event, callback):
                assert event in self._events
                self._event_callbacks[event].append(callback)
            def unsubscribe(self, event, callback):
                assert event in self._events
                try:
                    i = self._event_callbacks[event].index(callback)
                    self._event_callbacks[event] = self._event_callbacks[event][:i] + self._event_callbacks[event][i+1:]
                except:
                    pass
            def emit(self, event, data={}):
                assert event in self._events
                for cb in self._event_callbacks[event]:
                    cb(data)
        return EventfulWrap
