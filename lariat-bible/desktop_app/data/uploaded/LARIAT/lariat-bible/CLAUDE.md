# CLAUDE.md - The Lariat Bible

> **AI Assistant Guide for The Lariat Restaurant Management System**
> Last Updated: 2025-11-18

## Project Overview

**The Lariat Bible** is a comprehensive restaurant management system for The Lariat restaurant in Fort Collins, Colorado. This system serves as the single source of truth for vendor pricing, inventory management, recipe standardization, catering operations, equipment maintenance, and financial reporting.

### Key Business Metrics
- **Monthly Catering Revenue**: $28,000
- **Monthly Restaurant Revenue**: $20,000
- **Target Catering Margin**: 45%
- **Target Restaurant Margin**: 4%
- **Potential Annual Savings** (Shamrock vs SYSCO): $52,000 (~29.5% savings)

### Project Purpose
The owner, Sean, needs a data-driven system to:
1. Compare vendor pricing and identify savings (primarily Shamrock Foods vs SYSCO)
2. Track and cost recipes accurately
3. Optimize menu pricing to hit target margins
4. Manage equipment maintenance schedules
5. Streamline catering operations
6. Generate business intelligence reports

---

## Repository Structure

```
lariat-bible/
├── app.py                      # Flask web application entry point
├── setup.py                    # Initial setup script
├── requirements.txt            # Python dependencies
├── .env                        # Environment configuration (gitignored)
├── data/                       # Data storage
│   ├── invoices/              # Vendor invoices (images/PDFs)
│   └── *.csv                  # Sample comparison data
├── modules/                    # Business logic modules
│   ├── core/                  # Core integration
│   │   └── lariat_bible.py   # Main LariatBible class (359 lines)
│   ├── vendor_analysis/       # Price comparison & OCR
│   │   ├── __init__.py
│   │   ├── comparator.py     # VendorComparator class
│   │   ├── accurate_matcher.py
│   │   └── corrected_comparison.py
│   ├── recipes/               # Recipe management
│   │   └── recipe.py         # Recipe, Ingredient classes
│   ├── menu/                  # Menu items
│   │   ├── __init__.py
│   │   └── menu_item.py      # MenuItem class
│   ├── order_guides/          # Vendor order guides
│   │   └── order_guide_manager.py
│   ├── equipment/             # Equipment tracking
│   │   └── equipment_manager.py
│   └── email_parser/          # Email invoice parsing
│       └── email_parser.py
├── documentation/              # Additional docs
│   ├── README.md
│   ├── PRODUCT_MATCHING_VERIFICATION.md
│   ├── GITHUB_SETUP.md
│   └── GITHUB_QUICK_SETUP.md
└── CLAUDE.md                  # This file
```

### Lines of Code
- **Total Python Code**: ~2,422 lines
- **Development Stage**: Early/Active Development
- Many modules are scaffolded but not fully implemented

---

## Technology Stack

### Core Technologies
- **Language**: Python 3.8+
- **Web Framework**: Flask 3.0.0
- **Database**: SQLAlchemy 2.0.23 (SQLite for development)
- **API**: Flask-CORS, RESTful endpoints

### Data Processing
- **pandas** 2.1.4 - Data analysis and vendor comparisons
- **numpy** 1.26.2 - Numerical operations
- **openpyxl** 3.1.2 - Excel file handling

### OCR & Image Processing
- **pytesseract** 0.3.10 - OCR for invoice processing
- **Pillow** 10.1.0 - Image manipulation
- **opencv-python** 4.8.1.78 - Image preprocessing
- **PyPDF2** 3.0.1 - PDF processing

### Development Tools
- **black** - Code formatting (run before commits)
- **flake8** - Linting
- **pytest** - Testing framework
- **pre-commit** - Git hooks

### Other Key Libraries
- **python-dotenv** - Environment configuration
- **pydantic** - Data validation
- **rich** - Terminal output formatting
- **schedule** - Task scheduling (maintenance reminders)

---

## Core Architecture

### Main Integration Point: `LariatBible` Class

The `modules/core/lariat_bible.py` file contains the central `LariatBible` class that coordinates all modules:

```python
from modules.core.lariat_bible import lariat_bible

# Singleton instance available for import
lariat_bible.add_ingredient(ingredient)
lariat_bible.create_recipe_with_costing(recipe)
lariat_bible.run_comprehensive_comparison()
```

### Key Classes and Their Relationships

1. **Ingredient** (recipes/recipe.py)
   - Stores pricing from multiple vendors
   - Calculates best price and preferred vendor
   - Tracks last update timestamps

2. **Recipe** (recipes/recipe.py)
   - Contains RecipeIngredient objects
   - Calculates total cost and cost per portion
   - Analyzes vendor impact

3. **MenuItem** (menu/menu_item.py)
   - Links to Recipe
   - Tracks menu price and food cost
   - Calculates margins and suggests pricing

4. **VendorComparator** (vendor_analysis/comparator.py)
   - Compares Shamrock Foods vs SYSCO
   - Identifies savings opportunities
   - Generates reports

5. **OrderGuideManager** (order_guides/order_guide_manager.py)
   - Manages vendor product catalogs
   - Performs bulk price comparisons
   - Exports comparison spreadsheets

6. **EquipmentManager** (equipment/equipment_manager.py)
   - Tracks equipment and maintenance
   - Schedules preventive maintenance
   - Manages vendor contacts

---

## Development Workflows

### Setting Up the Development Environment

```bash
# Clone and setup
git clone <repository-url>
cd lariat-bible

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run setup
python setup.py

# Configure .env file
# Edit .env with your settings
```

### Running the Application

```bash
# Start Flask development server
python app.py

# Access at http://127.0.0.1:5000
# API endpoints:
# - GET /                       - Dashboard
# - GET /api/health            - Health check
# - GET /api/modules           - List modules
# - GET /api/vendor-comparison - Vendor savings
```

### Code Quality Standards

**Before committing:**
```bash
# Format code
black .

# Lint code
flake8 .

# Run tests
pytest
```

**Commit message format:**
```
<type>: <description>

Examples:
feat: Add invoice OCR processing
fix: Correct margin calculation in MenuItem
refactor: Simplify vendor comparison logic
docs: Update recipe costing documentation
test: Add tests for VendorComparator
```

---

## Key Conventions & Patterns

### 1. Vendor Names - CRITICAL
Always use consistent vendor names:
- **Shamrock Foods** (preferred vendor, ~29.5% cheaper)
- **SYSCO** (comparison vendor)

**Case sensitivity matters** in vendor comparisons!

### 2. Product Matching Rules

When comparing products between vendors:
- Match by EXACT product specification (not just name)
- Black Pepper Fine ≠ Black Pepper Coarse
- See `PRODUCT_MATCHING_VERIFICATION.md` for details
- Different grinds serve different culinary purposes

### 3. Pricing Calculations

**Margin formula:**
```python
margin = (menu_price - food_cost) / menu_price
```

**Target margins:**
- Catering: 45% (0.45)
- Restaurant: 4% (0.04)

**Suggested price:**
```python
suggested_price = food_cost / (1 - target_margin)
```

### 4. File Organization

**Data files:**
- Invoices: `data/invoices/`
- Exports: `data/exports/`
- Reports: `reports/`

**Naming conventions:**
- Use snake_case for Python files and variables
- Use descriptive names: `vendor_comparator` not `vc`
- Class names: PascalCase (e.g., `VendorComparator`)

### 5. Environment Variables

Required in `.env`:
```bash
DATABASE_URL=sqlite:///lariat.db
SECRET_KEY=<random-secret-key>
INVOICE_STORAGE_PATH=./data/invoices
RESTAURANT_NAME=The Lariat
PRIMARY_VENDOR=Shamrock Foods
COMPARISON_VENDOR=SYSCO
DEFAULT_CATERING_MARGIN=0.45
DEFAULT_RESTAURANT_MARGIN=0.04
```

---

## Module-Specific Guidelines

### Vendor Analysis Module

**Purpose**: Compare vendor prices, process invoices, identify savings

**Key files:**
- `comparator.py` - Main comparison logic
- `accurate_matcher.py` - Product matching algorithms
- Invoice processor (planned) - OCR for invoice data extraction

**When working on this module:**
- Always validate product matches are exact (grind, size, quality)
- Track price update timestamps
- Consider minimum order quantities
- Factor in delivery costs for true comparison

**Example usage:**
```python
from modules.vendor_analysis import VendorComparator

comparator = VendorComparator()
savings = comparator.compare_vendors('Shamrock Foods', 'SYSCO')
report = comparator.generate_report('reports/vendor_analysis.txt')
```

### Recipe Management Module

**Purpose**: Standardize recipes, calculate costs, analyze vendor impact

**Key classes:**
- `Ingredient` - Base ingredient with vendor pricing
- `RecipeIngredient` - Ingredient usage in recipe (quantity, unit)
- `Recipe` - Complete recipe with costing

**When working on this module:**
- Support recipe scaling (6 servings → 50 servings)
- Track cost history as ingredient prices change
- Consider prep time and labor costs (future)
- Handle fractional units properly (e.g., 0.25 tsp)

**Example usage:**
```python
from modules.recipes.recipe import Recipe, Ingredient, RecipeIngredient

# Create ingredient
flour = Ingredient(
    ingredient_id="ING001",
    name="All-Purpose Flour",
    sysco_price=18.99,
    shamrock_price=12.50,
    unit="LB"
)

# Create recipe
recipe = Recipe(
    recipe_id="REC001",
    name="Biscuits",
    category="Bakery",
    yield_amount=24,
    yield_unit="biscuits"
)
```

### Menu Management Module

**Purpose**: Link recipes to menu items, optimize pricing

**When working on this module:**
- Auto-update menu prices when recipe costs change
- Flag items below target margin
- Support different menu categories (catering vs restaurant)
- Calculate margin impact of price changes

### Equipment Management Module

**Purpose**: Track equipment, schedule maintenance

**When working on this module:**
- Use depreciation schedules
- Track repair history
- Alert on overdue maintenance
- Store vendor contact info

---

## Testing Strategy

### Test Organization
```
tests/
├── test_vendor_analysis.py
├── test_recipes.py
├── test_menu.py
└── test_integration.py
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific module
pytest tests/test_vendor_analysis.py

# Run with coverage
pytest --cov=modules --cov-report=html
```

### Test Data
- Use fixture data for consistent testing
- Don't commit real invoice data (sensitive)
- Mock vendor API calls if implemented

---

## Common Tasks for AI Assistants

### Adding a New Ingredient

```python
from modules.core.lariat_bible import lariat_bible
from modules.recipes.recipe import Ingredient

ingredient = Ingredient(
    ingredient_id="ING999",
    name="Paprika",
    category="Spices",
    sysco_price=15.99,
    sysco_case_size="6/1LB",
    shamrock_price=11.25,
    shamrock_case_size="25LB",
    unit="LB"
)

result = lariat_bible.add_ingredient(ingredient)
print(result)  # Shows preferred vendor and savings
```

### Creating a Recipe with Costing

```python
from modules.core.lariat_bible import lariat_bible
from modules.recipes.recipe import Recipe, RecipeIngredient

recipe = Recipe(
    recipe_id="REC999",
    name="BBQ Sauce",
    category="Sauce",
    yield_amount=1,
    yield_unit="gallon"
)

# Add ingredients
recipe.add_ingredient(RecipeIngredient(
    ingredient=existing_ingredient,
    quantity=2,
    unit="cups"
))

result = lariat_bible.create_recipe_with_costing(recipe)
print(result)  # Shows total cost, per-portion cost, vendor analysis
```

### Running Vendor Comparison

```python
from modules.core.lariat_bible import lariat_bible

# Import order guides
lariat_bible.import_order_guides(
    sysco_file="data/sysco_order_guide.xlsx",
    shamrock_file="data/shamrock_order_guide.xlsx"
)

# Run comprehensive comparison
results = lariat_bible.run_comprehensive_comparison()
print(f"Items compared: {results['items_compared']}")
print(f"Monthly savings: ${results['recommendations']['estimated_monthly_savings']}")
```

### Optimizing Menu Pricing

```python
from modules.core.lariat_bible import lariat_bible

# Get pricing optimization suggestions
optimizations = lariat_bible.optimize_menu_pricing()

for item in optimizations[:5]:  # Top 5
    print(f"{item['item']}: ${item['current_price']} → ${item['suggested_price']}")
    print(f"  Current margin: {item['current_margin']:.1%}")
    print(f"  Target margin: {item['target_margin']:.1%}")
```

### Generating Reports

```python
from modules.core.lariat_bible import lariat_bible

# Executive summary
summary = lariat_bible.generate_executive_summary()
print(summary)

# Vendor comparison report
from modules.vendor_analysis import VendorComparator
comparator = VendorComparator()
report = comparator.generate_report('reports/vendor_report.txt')

# Export all data
exports = lariat_bible.export_all_data('data/exports')
print(f"Exported files: {exports}")
```

---

## Important Business Context

### The Vendor Savings Discovery

Sean discovered that **Shamrock Foods offers ~29.5% better pricing** than SYSCO on many items. This is a HUGE finding ($52,000 annual savings potential).

**Critical considerations:**
1. Not all products are available from both vendors
2. Product specifications must match exactly (e.g., pepper grind size)
3. Delivery minimums and frequencies differ
4. Quality may vary between vendors for some items
5. Existing vendor relationships and payment terms matter

### The Catering Focus

Catering is the profit driver (45% margin vs 4% restaurant margin):
- Monthly catering revenue: $28,000
- Monthly restaurant revenue: $20,000
- **Strategy**: Focus on growing catering, maintain restaurant as marketing

### Equipment Tracking Matters

The restaurant has significant equipment investments. Proper maintenance:
- Prevents costly emergency repairs
- Extends equipment lifespan
- Maintains food safety compliance
- Reduces downtime

---

## Known Issues & Future Work

### Current Limitations

1. **Invoice OCR** - Planned but not fully implemented
   - Manual data entry currently required
   - Need to handle various invoice formats

2. **Database** - Using SQLite for development
   - Plan migration to PostgreSQL for production
   - Need proper migrations (Alembic configured)

3. **Authentication** - Basic setup only
   - No multi-user support yet
   - No role-based access control

4. **Mobile Interface** - Not implemented
   - Current focus is web interface
   - Need responsive design for kitchen staff

5. **Inventory Tracking** - Partially implemented
   - No real-time stock updates
   - No integration with POS system

### Planned Features

**Phase 1** (Current):
- ✅ Project structure
- ⏳ Vendor analysis core features
- ⏳ Recipe database

**Phase 2**:
- Invoice OCR pipeline
- Automated price tracking
- Email parsing for vendor invoices

**Phase 3**:
- Inventory tracking system
- Automated reorder points
- Integration with existing POS

**Phase 4**:
- Mobile-responsive dashboard
- Real-time reporting
- Staff training modules

---

## Security & Data Privacy

### Sensitive Data

**Never commit:**
- Actual invoice files (contain pricing, terms)
- `.env` file (contains secrets)
- Database files (contain business data)
- Export files with real business data

**Gitignored paths:**
- `data/invoices/*.jpg`, `*.png`, `*.pdf`
- `reports/*.xlsx`, `*.csv`, `*.pdf`
- `*.db`, `*.sqlite`, `*.sqlite3`
- `.env`, `.env.local`

### API Keys & Credentials

If vendor APIs are added:
- Store API keys in `.env`
- Never hardcode credentials
- Use environment variables
- Rotate keys regularly

---

## AI Assistant Guidelines

### When Working on This Codebase

1. **Understand the Business Context**
   - This is a real restaurant with real financial impact
   - Pricing accuracy matters ($52K in savings is significant)
   - Sean relies on this data for business decisions

2. **Be Careful with Calculations**
   - Margin calculations affect menu pricing
   - Vendor comparisons must be accurate
   - Test calculations with known examples

3. **Maintain Code Quality**
   - Run `black` and `flake8` before committing
   - Write docstrings for new functions
   - Add tests for new features
   - Update this CLAUDE.md when architecture changes

4. **Ask for Clarification**
   - Product matching rules (grind, quality, size)
   - Business logic (Why 45% vs 4% margin?)
   - Data sources (Where does this number come from?)

5. **Preserve Existing Patterns**
   - Follow established naming conventions
   - Use existing classes (extend, don't replace)
   - Maintain module separation
   - Keep the LariatBible class as integration point

6. **Document Decisions**
   - Add comments for complex business logic
   - Update docs when changing behavior
   - Note assumptions in code
   - Track TODOs with issue tracking

### Common Pitfalls to Avoid

❌ **Don't:**
- Compare products with different specifications
- Hardcode business metrics (use config)
- Skip input validation (garbage in = garbage out)
- Ignore unit conversions (pounds vs cases)
- Break existing APIs without migration path

✅ **Do:**
- Validate vendor names before comparison
- Check product matches are exact
- Handle edge cases (zero prices, missing data)
- Provide clear error messages
- Test with realistic data

---

## Quick Reference

### Import Paths
```python
# Main integration
from modules.core.lariat_bible import lariat_bible

# Vendor analysis
from modules.vendor_analysis import VendorComparator

# Recipes
from modules.recipes.recipe import Recipe, Ingredient, RecipeIngredient

# Menu items
from modules.menu.menu_item import MenuItem

# Order guides
from modules.order_guides.order_guide_manager import OrderGuideManager

# Equipment
from modules.equipment.equipment_manager import EquipmentManager
```

### Configuration Files
- `.env` - Environment variables
- `requirements.txt` - Python dependencies
- `.gitignore` - Files to exclude from git

### Data Directories
- `data/invoices/` - Vendor invoices
- `data/exports/` - Generated exports
- `reports/` - Business reports
- `logs/` - Application logs
- `backups/` - Database backups

### Useful Commands
```bash
# Run app
python app.py

# Setup project
python setup.py

# Format code
black .

# Lint code
flake8 .

# Run tests
pytest

# Install dependencies
pip install -r requirements.txt
```

---

## Getting Help

### Documentation
- **README.md** - Project overview and quick start
- **PRODUCT_MATCHING_VERIFICATION.md** - Vendor comparison rules
- **GITHUB_SETUP.md** - Git workflow
- **This file (CLAUDE.md)** - Comprehensive AI assistant guide

### Contact
- **Owner**: Sean
- **Restaurant**: The Lariat, Fort Collins, CO

### When Something Breaks
1. Check `.env` configuration
2. Verify virtual environment is activated
3. Ensure dependencies are installed
4. Check logs in `logs/` directory
5. Review recent git commits

---

## Changelog

### 2025-11-18
- Initial CLAUDE.md creation
- Documented current codebase structure (~2,422 lines)
- Established conventions and guidelines
- Added comprehensive examples and workflows

---

**Remember**: This system helps a real business make real decisions. Accuracy, clarity, and reliability are paramount. When in doubt, ask questions and verify assumptions.

Happy coding! 🤠
