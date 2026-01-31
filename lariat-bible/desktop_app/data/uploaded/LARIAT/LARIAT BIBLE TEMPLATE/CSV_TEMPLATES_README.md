
# CSV Templates for The Lariat Bible

This folder contains CSV templates for managing restaurant operations. All templates are ready to use with the Lariat Bible application.

## Template Files

### 1. **vendor_product_comparison_template.csv**
Compare product pricing between vendors (Shamrock vs SYSCO).

**Columns:**
- `product` - Product name with specification
- `sysco_per_lb` - SYSCO price per pound
- `shamrock_per_lb` - Shamrock price per pound
- `savings_per_lb` - Savings amount per pound
- `savings_percent` - Percentage savings
- `preferred_vendor` - Which vendor is cheaper
- `sysco_split_per_lb` - SYSCO split case price per pound (optional)
- `split_vs_shamrock` - Split case vs Shamrock comparison (optional)

**Use Case:** Quick vendor price comparison for decision making

---

### 2. **spice_comparison_template.csv**
Detailed spice/ingredient comparison with pack sizes and estimated savings.

**Columns:**
- `item` - Item name
- `sysco_pack` - SYSCO package size (e.g., "6/1LB")
- `sysco_case_price` - SYSCO case price
- `sysco_per_lb` - SYSCO price per pound
- `sysco_split_price` - SYSCO split case price (optional)
- `sysco_split_per_lb` - Split price per pound (optional)
- `shamrock_pack` - Shamrock package size
- `shamrock_case_price` - Shamrock case price
- `shamrock_per_lb` - Shamrock price per pound
- `savings_per_lb` - Savings per pound
- `savings_percent` - Percentage savings
- `preferred_vendor` - Preferred vendor
- `monthly_savings_estimate` - Estimated monthly savings

**Use Case:** Comprehensive ingredient cost analysis

---

### 3. **invoice_template.csv**
Track vendor invoices and purchases.

**Columns:**
- `date` - Invoice date (YYYY-MM-DD)
- `vendor` - Vendor name
- `invoice_number` - Invoice number
- `item_code` - Vendor item code
- `item_description` - Item description
- `pack_size` - Package size
- `quantity` - Quantity ordered
- `unit_price` - Price per unit
- `total_price` - Total line item price
- `category` - Product category

**Use Case:** Invoice tracking and cost analysis

---

### 4. **order_guide_template.csv**
Master order guide with all available products from vendors.

**Columns:**
- `vendor` - Vendor name
- `category` - Product category
- `item_code` - Vendor item code
- `item_name` - Product name
- `pack_size` - Package size
- `case_price` - Case price
- `split_price` - Split case price (if available)
- `unit_price_per_lb` - Price per pound
- `notes` - Additional notes
- `last_updated` - Last price update date

**Use Case:** Central product catalog for ordering

---

### 5. **recipe_cost_template.csv**
Calculate recipe costs and portions.

**Columns:**
- `recipe_name` - Recipe/dish name
- `ingredient` - Ingredient name
- `quantity` - Quantity needed
- `unit` - Unit of measure (lb, oz, gal, etc.)
- `cost_per_unit` - Cost per unit
- `total_cost` - Total ingredient cost
- `yield_servings` - Number of servings produced
- `cost_per_serving` - Cost per serving
- `category` - Ingredient category

**Use Case:** Recipe costing and menu pricing

---

### 6. **inventory_template.csv**
Track current inventory levels and values.

**Columns:**
- `item_name` - Item name
- `category` - Product category
- `current_stock` - Current quantity in stock
- `unit` - Unit of measure
- `par_level` - Target inventory level
- `reorder_point` - When to reorder
- `preferred_vendor` - Preferred vendor
- `last_order_date` - Last order date
- `cost_per_unit` - Current cost per unit
- `total_value` - Total inventory value

**Use Case:** Inventory management and ordering

---

### 7. **catering_event_template.csv**
Track catering events and profitability.

**Columns:**
- `event_date` - Event date
- `event_name` - Event name
- `client_name` - Client name
- `guest_count` - Number of guests
- `menu_selection` - Menu chosen
- `total_cost` - Total event cost
- `sale_price` - Price charged to client
- `profit` - Profit amount
- `profit_margin` - Profit margin percentage
- `status` - Event status (Confirmed, Pending, Quote Sent, etc.)

**Use Case:** Catering sales and profitability tracking

---

### 8. **equipment_maintenance_template.csv**
Schedule and track equipment maintenance.

**Columns:**
- `equipment_name` - Equipment name
- `equipment_type` - Type of equipment
- `location` - Equipment location
- `last_service_date` - Last service date
- `next_service_date` - Next scheduled service
- `service_interval_days` - Days between services
- `service_provider` - Service company name
- `last_service_cost` - Cost of last service
- `notes` - Service notes
- `status` - Current status (Good, Needs Service, Due Soon)

**Use Case:** Preventive maintenance scheduling

---

## How to Use These Templates

### Method 1: Direct Import to Application
1. Copy template to `lariat-bible/data/` folder
2. Rename if needed
3. Application will auto-detect and load the CSV

### Method 2: Via API
```bash
# List all CSV files
curl http://127.0.0.1:5000/api/data/list-csv-files

# Load specific comparison
curl http://127.0.0.1:5000/api/data/vendor-comparison-csv
```

### Method 3: Python Script
```python
from modules.data_loader import CSVLoader

loader = CSVLoader()
df = loader.load_csv('your_template.csv')
print(df.head())
```

## Tips for Using Templates

### 1. Data Entry
- Keep dates in YYYY-MM-DD format
- Use consistent vendor names (exact spelling)
- Include units for all quantities
- Leave cells empty if data not available (don't use "N/A" or "-")

### 2. Price Calculations
- Per pound prices are automatically calculated
- Savings percentages are computed automatically
- Monthly estimates based on usage patterns

### 3. Categories
Use consistent category names:
- Spices
- Protein
- Produce
- Dry Goods
- Canned Goods
- Oil & Vinegar
- Dairy
- Frozen
- Supplies

### 4. Pack Size Format
Standard formats:
- `6/1LB` = 6 containers of 1 pound each
- `25 LB` = 25 pound bag/box
- `1/6/LB` = 1 container of 6 pounds
- `24 CT` = 24 count
- `#10` = #10 can size

## Integration with Application

All templates work with the test suite:
```bash
# Run tests to verify CSV loading
python3 -m pytest tests/test_csv_loader.py -v
```

## Customization

Feel free to:
- Add new columns for your specific needs
- Create additional templates
- Modify sample data
- Add more products/recipes

The CSV loader is flexible and will adapt to your column structure.

## Getting Started

1. **Start Simple**: Begin with vendor_product_comparison_template.csv
2. **Add Real Data**: Replace sample data with your actual prices
3. **Expand**: Add more templates as needed
4. **Automate**: Use API endpoints to access data programmatically

## Support

For questions about templates:
- See `TESTING_SUMMARY.md` for technical details
- See `QUICK_START.md` for API usage
- See `tests/README.md` for testing examples

## Template Updates

These templates are based on current data structure. Update them as your needs evolve:
- Add new vendors as columns
- Create category-specific templates
- Build specialized reports

---

**Last Updated:** 2025-11-18
**Version:** 1.0
**Compatible with:** The Lariat Bible v1.0
