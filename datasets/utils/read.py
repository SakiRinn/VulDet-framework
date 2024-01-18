

def read_text(path):
    lines = []
    with open(path, 'r') as f:
        lines = f.readlines()
    return ' '.join(lines)

def read_csv(path, delimiter='\t'):
    data = []
    with open(path, 'r') as f:
        header = f.readline().strip()
        header_parts = [hp.strip() for hp in header.split(delimiter)]
        for line in f:
            line = line.strip()
            line_parts = line.split(delimiter)

            instance = {}
            for i, hp in enumerate(header_parts):
                content = line_parts[i].strip() if i < len(line_parts) else ''
                instance[hp] = content
            data.append(instance)
    return data

def read_code(path):
    code_lines = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if '//' in line:
                line = line[:line.index('//')]
            code_lines[i + 1] = line
        return code_lines
