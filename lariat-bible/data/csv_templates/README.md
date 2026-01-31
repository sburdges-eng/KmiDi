# CSV Templates Integration

All standard templates are now available in `data/csv_templates/` for use with the Lariat Bible app. See `CSV_TEMPLATES_README.md` for details.

To load a template in Python:
```python
import pandas as pd
path = 'data/csv_templates/vendor_product_comparison_template.csv'
df = pd.read_csv(path)
print(df.head())
```
