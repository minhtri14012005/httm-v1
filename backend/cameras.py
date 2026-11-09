import sys


def enumerate_devices():
    """Enumerate DirectShow names without opening any camera."""
    if sys.platform != 'win32':
        raise RuntimeError('Tự dò camera hiện hỗ trợ Windows DirectShow; máy khác cần adapter')
    import comtypes
    comtypes.CoInitialize()
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        names = graph.get_input_devices()
        del graph
        return names
    finally:
        comtypes.CoUninitialize()


def slots(config):
    try:
        names = enumerate_devices()
    except Exception as exc:
        return [{'key':k, 'label':label, 'available':False, 'error':str(exc)}
                for k,label in [('laptop','Webcam Laptop'),('camo','Camo Camera')]]
    result = []
    for key,label in [('laptop','Webcam Laptop'),('camo','Camo Camera')]:
        exact = config['camera_names'].get(key)
        if exact:
            candidates = [(i,n) for i,n in enumerate(names) if n == exact]
        elif key == 'camo':
            candidates = [(i,n) for i,n in enumerate(names) if 'camo' in n.lower()]
        else:
            hints = ['integrated', 'built-in', 'easycamera', 'truevision', 'hd webcam', 'hd camera', 'usb2.0 hd uvc', 'fhd camera', 'user facing']
            candidates = [(i,n) for i,n in enumerate(names) if 'camo' not in n.lower() and any(h in n.lower() for h in hints)]
        if len(candidates)==1:
            i,n = candidates[0]
            result.append({'key':key, 'label':label, 'available':True, 'name':n, 'index':i})
        else:
            result.append({'key':key, 'label':label, 'available':False,
                           'error':'Không tìm thấy hoặc tên camera không rõ ràng; cấu hình camera_names bằng tên DirectShow chính xác'})
    return result
