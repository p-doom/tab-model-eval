# Assistant <NO_EVAL>
```bash
cat -n src/data_loader.py
```

# User
<stdout>
     1  import os
     2  import json
     3  from pathlib import Path
     4  from typing import List, Dict, Optional
     5  
     6  class DataLoader:
     7      def __init__(self, data_dir: str):
     8          self.data_dir = Path(data_dir)
     9          self.cache: Dict[str, any] = {}
    10  
    11      def load_json(self, filename: str) -> Dict:
    12          filepath = self.data_dir / filename
    13          with open(filepath, "r") as f:
    14              return json.load(f)
    15  
    16      def load_all_files(self, extension: str = ".json") -> List[Dict]:
    17          results = []
    18          for file in self.data_dir.glob(f"*{extension}"):
    19              results.append(self.load_json(file.name))
    20          return results
    21  
    22      def get_cached_or_load(self, filename: str) -> Dict:
    23          if filename not in self.cache:
    24  
    25          return self.cache[filename]
    26  
    27      def filter_by_key(self, data: List[Dict], key: str, value: any) -> List[Dict]:
    28  
    29  
    30      def save_json(self, data: Dict, filename: str) -> None:
    31          filepath = self.data_dir / filename
    32          with open(filepath, "w") as f:
    33  
</stdout>

# Assistant <NO_EVAL>
```bash
sed -i '24,24c\            self.cache[filename] = self.load_' src/data_loader.py && cat -n src/data_loader.py | sed -n '19,28p'
```

# User
<stdout>
    19              results.append(self.load_json(file.name))
    20          return results
    21  
    22      def get_cached_or_load(self, filename: str) -> Dict:
    23          if filename not in self.cache:
    24              self.cache[filename] = self.load_
    25          return self.cache[filename]
    26  
    27      def filter_by_key(self, data: List[Dict], key: str, value: any) -> List[Dict]:
    28  
</stdout>

# Assistant <EVAL>
```bash
sed -i '24,24c\            self.cache[filename] = self.load_json(filename)' src/data_loader.py && cat -n src/data_loader.py | sed -n '19,28p'
```

# User
<stdout>
    19              results.append(self.load_json(file.name))
    20          return results
    21  
    22      def get_cached_or_load(self, filename: str) -> Dict:
    23          if filename not in self.cache:
    24              self.cache[filename] = self.load_json(filename)
    25          return self.cache[filename]
    26  
    27      def filter_by_key(self, data: List[Dict], key: str, value: any) -> List[Dict]:
    28  
</stdout>

# Assistant <NO_EVAL>
```bash
sed -i '28,28c\        return [item for item in data if item.get(' src/data_loader.py && cat -n src/data_loader.py | sed -n '23,32p'
```

# User
<stdout>
    23          if filename not in self.cache:
    24              self.cache[filename] = self.load_json(filename)
    25          return self.cache[filename]
    26  
    27      def filter_by_key(self, data: List[Dict], key: str, value: any) -> List[Dict]:
    28          return [item for item in data if item.get(
    29  
    30      def save_json(self, data: Dict, filename: str) -> None:
    31          filepath = self.data_dir / filename
    32          with open(filepath, "w") as f:
</stdout>

# Assistant <EVAL>
```bash
sed -i '28,28c\        return [item for item in data if item.get(key) == value]' src/data_loader.py && cat -n src/data_loader.py | sed -n '23,32p'
```

# User
<stdout>
    23          if filename not in self.cache:
    24              self.cache[filename] = self.load_json(filename)
    25          return self.cache[filename]
    26  
    27      def filter_by_key(self, data: List[Dict], key: str, value: any) -> List[Dict]:
    28          return [item for item in data if item.get(key) == value]
    29  
    30      def save_json(self, data: Dict, filename: str) -> None:
    31          filepath = self.data_dir / filename
    32          with open(filepath, "w") as f:
</stdout>

# Assistant <NO_EVAL>
```bash
sed -i '33,33c\            json.dump(data, f, indent=' src/data_loader.py && cat -n src/data_loader.py | sed -n '26,35p'
```

# User
<stdout>
    26  
    27      def filter_by_key(self, data: List[Dict], key: str, value: any) -> List[Dict]:
    28          return [item for item in data if item.get(key) == value]
    29  
    30      def save_json(self, data: Dict, filename: str) -> None:
    31          filepath = self.data_dir / filename
    32          with open(filepath, "w") as f:
    33              json.dump(data, f, indent=
    34  
    35  
</stdout>

# Assistant <EVAL>
```bash
sed -i '33,33c\            json.dump(data, f, indent=2)' src/data_loader.py && cat -n src/data_loader.py | sed -n '26,35p'
```

# User
<stdout>
    26  
    27      def filter_by_key(self, data: List[Dict], key: str, value: any) -> List[Dict]:
    28          return [item for item in data if item.get(key) == value]
    29  
    30      def save_json(self, data: Dict, filename: str) -> None:
    31          filepath = self.data_dir / filename
    32          with open(filepath, "w") as f:
    33              json.dump(data, f, indent=2)
    34  
    35  
</stdout>
