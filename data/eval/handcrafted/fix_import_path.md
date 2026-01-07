# Assistant <NO_EVAL>
```bash
python src/main.py
```

# User
<stdout>
Traceback (most recent call last):
  File "src/main.py", line 6, in <module>
    from utils import load_config
ImportError: cannot import name 'load_config' from 'utils'
</stdout>

# Assistant <NO_EVAL>
```bash
cat -n src/common/config_utils.py
```

# User
<stdout>
     1  import yaml
     2  import os
     3  
     4  def load_config(path):
     5      if not os.path.exists(path):
     6          return {}
     7      with open(path, 'r') as f:
     8          return yaml.safe_load(f)
     9  
    10  def save_config(config, path):
    11      with open(path, 'w') as f:
    12          yaml.dump(config, f)
</stdout>

# Assistant <NO_EVAL>
```bash
cat -n src/main.py
```

# User
<stdout>
     1  import os
     2  import sys
     3  import time
     4  import logging
     5  
     6  from utils import load_config
     7  
     8  def main():
     9      setup_logging()
    10      config = load_config("config.yaml")
    11      print(f"Loaded config: {config}")
    12  
    13  if __name__ == "__main__":
    14      main()
</stdout>


# Assistant <EVAL>
```bash
sed -i '6,6c\from common.config_utils import load_config' src/main.py && cat -n src/main.py | sed -n '1,10p'
```

# User
<stdout>
     1  import os
     2  import sys
     3  import time
     4  import logging
     5  
     6  from common.config_utils import load_config
     7  
     8  def main():
     9      setup_logging()
    10      config = load_config("config.yaml")
</stdout>
