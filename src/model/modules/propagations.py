import torch


def get_propagations(config):
    propagation_type = config.get('propagation', 'window')
    window_size = config['window_size']
    dilated = config.get('dilated', 1)
    assert window_size % 2 == 1
    full_coords = None

    # [L,2] (y,x)
    if propagation_type == 'window':
        coords = torch.stack(torch.meshgrid(torch.arange(-window_size // 2 + 1, window_size // 2 + 1),
                                            torch.arange(-window_size // 2 + 1, window_size // 2 + 1)), dim=-1)
        coords = coords.reshape(-1, 2)
    elif propagation_type == 'dilated1':  # square
        assert dilated > 1
        coords = [[0, 0]]
        for w in range(0, window_size // 2 + 1):
            for j in range(0, window_size // 2 + 1):
                if w + j == 0:
                    continue
                coords.append([dilated * j, dilated * w])
                if w != 0:
                    coords.append([dilated * j, -dilated * w])
                if j != 0:
                    coords.append([-dilated * j, dilated * w])
                if w != 0 and j != 0:
                    coords.append([-dilated * j, -dilated * w])
        coords = torch.tensor(coords, dtype=torch.long)

        # full coords
        full_coords = [[0, 0]]
        for w in range(0, window_size // 2 * dilated + 1):
            for j in range(0, window_size // 2 * dilated + 1):
                if w + j == 0:
                    continue
                full_coords.append([j, w])
                if w != 0:
                    full_coords.append([j, -w])
                if j != 0:
                    full_coords.append([-j, w])
                if w != 0 and j != 0:
                    full_coords.append([-j, -w])
        full_coords = torch.tensor(full_coords, dtype=torch.long)
    elif propagation_type == 'dilated2':  # rhombus
        raise NotImplementedError('Not implemented propagation type!')
    elif propagation_type == 'topk': # in topk window weights are useless
        coords = torch.stack(torch.meshgrid(torch.arange(-window_size // 2 + 1, window_size // 2 + 1),
                                            torch.arange(-window_size // 2 + 1, window_size // 2 + 1)), dim=-1)
        coords = coords.reshape(-1, 2)
    else:
        raise NotImplementedError('Not implemented propagation type!')
    return coords, full_coords
