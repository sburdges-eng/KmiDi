# Toast POS Integration Guide for The Lariat

> **Comprehensive documentation of Toast POS CSV templates, data exports, API schemas, and integration patterns**
>
> Research Date: 2025-11-18
> Toast Platform Documentation: doc.toasttab.com
> Toast Central Support: central.toasttab.com

---

## Table of Contents

1. [Overview](#overview)
2. [CSV Import Templates](#csv-import-templates)
3. [Data Export Formats](#data-export-formats)
4. [API Structure](#api-structure)
5. [Integration with The Lariat](#integration-with-the-lariat)
6. [Template Examples](#template-examples)
7. [Best Practices](#best-practices)

---

## Overview

Toast POS is an industry-leading restaurant management platform offering:
- Menu management via CSV bulk import/export
- Automated nightly data exports (sales, payments, orders, labor)
- RESTful API (Menus v2/v3)
- Catering & Events module with BEO support
- Inventory and recipe costing integration

**Toast Modules Relevant to The Lariat:**
- ✅ **Menus** - Menu items, modifiers, pricing
- ✅ **Catering & Events** - BEO management, prep tools
- ✅ **Data Exports** - Sales, orders, payments in CSV
- ✅ **Recipe Costing** (xtraCHEF) - Ingredient tracking, cost analysis
- ✅ **Reports** - Revenue, item sales, performance metrics

---

## CSV Import Templates

### Accessing Templates

**Location:** Toast Web → Menus → Bulk Management → Bulk Import Tool

**Format:** Google Sheets templates (export to CSV for upload)

**Three Template Types:**
1. **Basic Template** - Quick menu item creation
2. **Item Update Template** - Update existing items
3. **Advanced Template** - Full control with advanced settings

---

### 1. Basic Template

**Purpose:** Quickly create menu items, modifier groups, and modifiers

**Required Columns:**

| Column | Type | Required | Values/Format |
|--------|------|----------|---------------|
| **Operation** | Dropdown | Yes | Always `CREATE` |
| **Entity type** | Dropdown | Yes | `MENU_ITEM`, `MODIFIER_GROUP`, `MODIFIER` |
| **Operation ID** | String (max 255) | Yes | Unique identifier (1, 2, 3...) |
| **Name** | String (max 255) | Yes | Item name |
| **Parent entity type** | Dropdown | Yes | `MENU_GROUP`, `MENU_ITEM`, `MODIFIER_GROUP` |
| **Parent version ID or operation ID** | String | Yes | Toast GUID or operation ID |
| **Pricing strategy or method** | Dropdown | Yes | `BASE` (fixed price) |
| **Price** | Numeric string | No | Format: "10.00" or "10" |

**Example CSV:**
```csv
Operation,Entity type,Operation ID,Name,Parent entity type,Parent version ID or operation ID,Pricing strategy or method,Price
CREATE,MENU_ITEM,1,Green Chile Bowl,MENU_GROUP,500000000032822323,BASE,12.99
CREATE,MENU_ITEM,2,BBQ Pork Sandwich,MENU_GROUP,500000000032822323,BASE,10.50
CREATE,MODIFIER_GROUP,3,Protein Options,MENU_ITEM,1,PRICED_BY_MODIFIERS,
CREATE,MODIFIER,4,Add Chicken,MODIFIER_GROUP,3,BASE,3.00
CREATE,MODIFIER,5,Add Pork,MODIFIER_GROUP,3,BASE,4.00
```

**Price Validation:**
- ✅ Valid: `"10.00"`, `"10"`, `".99"`, `"-.50"`, `"(2.00)"`
- ❌ Invalid: `"$10.00"`, `"10.00.00"`, `"ten"`

**Constraints:**
- All items must use `BASE` pricing strategy
- Cannot set advanced options (SKU, PLU, visibility)
- Fastest for simple menu creation

---

### 2. Item Update Template

**Purpose:** Update existing menu items (name, price, POS name, kitchen name, description, PLU, SKU)

**Required Columns:**

| Column | Type | Required | Notes |
|--------|------|----------|-------|
| **Operation** | Dropdown | Yes | Always `UPDATE` |
| **Entity type** | Dropdown | Yes | Always `MENU_ITEM` |
| **Operation ID** | String (max 255) | Yes | Unique identifier |
| **Version ID or operation ID** | String | Yes | Toast GUID of target item |
| **Name** | String (max 255) | No | Updated name |
| **POS name** | String (max 255) | No | Display name on POS |
| **Kitchen name** | String (max 255) | No | Kitchen ticket display |
| **Item description** | String (max 1000) | No | Description |
| **Price** | Numeric string | No | BASE price only |
| **PLU** | String (max 255) | No | Price look-up code |
| **SKU** | String (max 255) | No | Stock-keeping unit |
| **Sales category multiLocation ID** | String | No | Toast GUID |
| **Prep station multiLocation IDs** | Comma-separated | No | Multiple GUIDs allowed |
| **Tax rate multiLocation IDs** | Comma-separated | No | Multiple GUIDs allowed |
| **Guest count** | Decimal | No | Rounds to 2 places |

**Example CSV:**
```csv
Operation,Entity type,Operation ID,Version ID or operation ID,Name,Price,PLU,SKU
UPDATE,MENU_ITEM,1,600000000045678901,Green Chile Bowl - LARGE,14.99,1001,GCB-L
UPDATE,MENU_ITEM,2,600000000045678902,BBQ Pork Sandwich,11.50,1002,BBQ-P
```

**Important Rules:**
- ⚠️ **Blank cells retain original data** - Cannot clear existing values
- Only works with `BASE` pricing strategy items
- Must provide Toast GUID (version ID) to identify target item
- Changes apply immediately upon upload

---

### 3. Advanced Template

**Purpose:** Full control over menu creation/update/attachment with all advanced settings

**Core Columns (All Operations):**

| Column | Type | Required | Values |
|--------|------|----------|--------|
| **Operation** | Dropdown | Yes | `CREATE`, `UPDATE`, `ATTACH` |
| **Entity type** | Dropdown | Yes | `MENU_ITEM`, `MODIFIER_GROUP`, `MODIFIER` |
| **Operation ID** | String (max 255) | Yes | Unique identifier |
| **Parent entity type** | Dropdown | Yes | `MENU_GROUP`, `MENU_ITEM`, `MODIFIER_GROUP` |
| **Parent version ID or operation ID** | String | Yes | Toast GUID or operation ID |

**CREATE Operation Additional Columns:**

| Column | Type | Required | Notes |
|--------|------|----------|-------|
| **Name** | String (max 255) | Yes | Entity name |
| **POS name** | String (max 255) | No | POS display name |
| **Kitchen name** | String (max 255) | No | Kitchen display name |
| **Item description** | String (max 1000) | No | Menu items only |
| **Button color** | Dropdown/Hex | No | See button colors below |
| **Pricing strategy or method** | Dropdown | Yes | `BASE`, `LOCATION_SPECIFIC`, `PRICED_BY_MODIFIERS` |
| **Price** | Numeric string | No | Required for BASE pricing |
| **Location-specific price target ID** | String | No | Required for LOCATION_SPECIFIC |
| **PLU** | String (max 255) | No | Price look-up code |
| **SKU** | String (max 255) | No | Stock-keeping unit |
| **Prep station multiLocation IDs** | Comma-separated | No | Kitchen routing |
| **Tax rate multiLocation IDs** | Comma-separated | No | Tax assignments |
| **Sales category multiLocation ID** | String | No | Reporting category |
| **Visible to POS** | Dropdown | No | `TRUE` or `FALSE` |
| **Visible to Kiosk, Toast Order and Pay** | Dropdown | No | `TRUE` or `FALSE` |
| **Visible to Toast Online Ordering, Toast Takeout App** | Dropdown | No | `TRUE` or `FALSE` |
| **Visible to online ordering partners** | Dropdown | No | `TRUE` or `FALSE` |
| **Contains alcohol** | Dropdown | No | `YES` or `NO` |
| **Guest count** | Decimal | No | For menu items |

**Modifier-Specific Columns:**

| Column | Type | Required | Notes |
|--------|------|----------|-------|
| **Modifier name** | String (max 255) | No | Modifiers only |
| **Modifier target ID** | String | No | Toast GUID |
| **Modifier owner ID** | String | No | Toast GUID |
| **Default modifier** | Dropdown | No | `TRUE` or `FALSE` |

**Button Color Options:**

```
WHITE (#ffffff)
RED_50 (#f27979), RED_75 (#e35350), RED_100 (#c92b28)
ORANGE_50 (#f3a376), ORANGE_75 (#ed7838), ORANGE_100 (#d75e1f)
YELLOW_50 (#f6d98a), YELLOW_75 (#f3c855), YELLOW_100 (#c29e15)
GREEN_50 (#90c490), GREEN_75 (#64ae64), GREEN_100 (#348734)
BLUE_50 (#7db9e8), BLUE_75 (#4699de), BLUE_100 (#107ed9)
PURPLE_50 (#b397d6), PURPLE_75 (#956bc7), PURPLE_100 (#7646b7)
GRAY_50 (#c5c5c5), GRAY_75 (#a3a3a3), GRAY_100 (#717171)
```

**Example Advanced CSV:**
```csv
Operation,Entity type,Operation ID,Name,Parent entity type,Parent version ID or operation ID,Pricing strategy or method,Price,PLU,SKU,Button color,Visible to POS,Contains alcohol
CREATE,MENU_ITEM,1,The Lariat Burger,MENU_GROUP,500000000032822323,BASE,13.99,2001,BURGER-01,RED_75,TRUE,NO
CREATE,MENU_ITEM,2,Craft Beer - IPA,MENU_GROUP,500000000032822324,BASE,7.00,3001,BEER-IPA,YELLOW_100,TRUE,YES
UPDATE,MENU_ITEM,3,600000000045678903,Green Chile Special,,,15.99,1003,GCH-SP,GREEN_75,TRUE,NO
ATTACH,MODIFIER_GROUP,4,,MENU_ITEM,1,600000000045678904,,,,,,
```

**Pricing Strategies:**

1. **BASE** - Fixed price (most common)
   - Example: "Burger - $12.99"

2. **LOCATION_SPECIFIC** - Different price per location
   - Requires: `Location-specific price target ID`
   - Use case: Chain restaurants with regional pricing

3. **PRICED_BY_MODIFIERS** - Price determined by modifiers
   - Modifier groups only
   - Example: "Pizza - Build Your Own" (base + toppings)

---

## Data Export Formats

### Export Configuration

**Access:** Toast Web → Reports → Settings → Data Exports

**Export Timing:**
- Automated: After closeout hour (default 4:00 AM)
- Manual: Download anytime from Toast Web
- Access methods: Web UI, CLI, FTP

**Configurable Options:**
- ✅ Select which data types to export
- ✅ Choose specific columns to include/exclude
- ✅ Reorder columns
- ✅ Set export schedule

---

### 1. OrderDetails.csv

**Contains:** Order-level information

**Key Fields:**

| Field Name | Type | Description |
|------------|------|-------------|
| **Location** | String | Restaurant location name |
| **Order Id** | Long | Unique order identifier |
| **Order #** | String | Display order number |
| **Checks** | Integer | Number of checks in order |
| **Opened** | DateTime | Order opened timestamp (MM/DD/YYYY HH:MM:SS) |
| **Paid** | DateTime | Payment timestamp |
| **Closed** | DateTime | Order closed timestamp |
| **Duration** | Integer | Minutes from opened to closed |
| **Discount Amount** | Currency | Total discounts (2 decimals) |
| **Amount** | Currency | Subtotal before tax/tip |
| **Tax** | Currency | Tax amount |
| **Tip** | Currency | Tip amount |
| **Gratuity** | Currency | Auto-gratuity amount |
| **Total** | Currency | Final total |
| **Server** | String | Server name |
| **Table** | String | Table number/name |
| **Revenue Center** | String | Revenue category |
| **Dining Area** | String | Physical dining area |
| **Service** | String | Daypart (Breakfast, Lunch, Dinner) |
| **# of Guests** | Integer | Guest count |
| **Voided** | Boolean | TRUE if voided |
| **Order Source** | String | POS, Kiosk, Online, etc. |

**Example Row:**
```csv
Location,Order Id,Order #,Opened,Paid,Closed,Amount,Tax,Tip,Total,Server,Table,# of Guests,Voided,Order Source
The Lariat,987654321012345,1234,11/15/2025 12:34:56,11/15/2025 12:45:30,11/15/2025 12:46:00,45.50,3.64,9.00,58.14,Jane Smith,12,4,FALSE,POS
```

**Use Cases for The Lariat:**
- Daily revenue tracking ($28K catering + $20K restaurant target)
- Server performance analysis
- Peak hour identification
- Average order value calculation

---

### 2. CheckDetails.csv

**Contains:** Individual check information

**Key Fields:**

| Field Name | Type | Description |
|------------|------|-------------|
| **Check Id** | Long | Unique check identifier |
| **Check #** | String | Display check number |
| **Opened Date** | Date | Check opened date |
| **Opened Time** | Time | Check opened time |
| **Total** | Currency | Check total |
| **Tax** | Currency | Tax amount |
| **Discount** | Currency | Discount amount |
| **Reason of Discount** | String | Discount reason/code |
| **Item Description** | String | Comma-separated list of items |
| **Customer Id** | String | Customer identifier (if logged in) |
| **Customer Name** | String | Customer name |
| **Customer Phone** | String | Phone number |
| **Customer Email** | String | Email address |
| **Server** | String | Server name |
| **Table Size** | Integer | Number of seats |
| **Tender** | String | Payment method |
| **Link to check image** | URL | URL to check image (if available) |

**Example Row:**
```csv
Check Id,Check #,Opened Date,Total,Item Description,Customer Name,Customer Phone,Server,Table Size
123456789012345,5001,11/15/2025,28.50,"Green Chile Bowl, Iced Tea, Side Salad",John Doe,970-555-1234,Sarah Johnson,2
```

**Use Cases for The Lariat:**
- Customer order history tracking
- Repeat customer identification
- Check average analysis
- Upsell opportunity identification

---

### 3. ItemSelectionDetails.csv

**Contains:** Individual item orders (most detailed)

**Key Fields:**

| Field Name | Type | Description |
|------------|------|-------------|
| **Menu** | String | Top-level menu |
| **Menu Group** | String | Menu category |
| **Menu Subgroup** | String | Subcategory |
| **Menu Item** | String | Item name |
| **SKU** | String | Stock-keeping unit |
| **Gross Price** | Currency | Price before discounts |
| **Discount** | Currency | Discount applied |
| **Net Price** | Currency | Price after discounts |
| **Tax** | Currency | Tax amount |
| **Qty** | Integer | Quantity ordered |
| **Void Qty** | Integer | Quantity voided |
| **Item Qty** | Integer | Net quantity (Qty - Void Qty) |
| **% of total orders** | Decimal (5 places) | Percentage of all orders |
| **% qty by group** | Decimal | % within menu group |
| **% qty by menu** | Decimal | % within menu |
| **% qty by all** | Decimal | % of all items |
| **% net amount by group** | Decimal | Revenue % within group |
| **% net amount by menu** | Decimal | Revenue % within menu |
| **% net amount by all** | Decimal | Revenue % of all items |
| **Void?** | Boolean | TRUE if voided |
| **Deferred** | Boolean | TRUE if deferred item |
| **Tax Exempt** | Boolean | TRUE if tax exempt |
| **Tax Inclusion Option** | String | Tax calculation method |

**Example Row:**
```csv
Menu,Menu Group,Menu Item,SKU,Gross Price,Net Price,Qty,Void Qty,Item Qty,% of total orders,Void?
Lunch Menu,Entrees,Green Chile Bowl,GCB-REG,12.99,12.99,45,2,43,8.25000,FALSE
```

**Use Cases for The Lariat:**
- **Menu engineering** - Identify stars, plowhorses, puzzles, dogs
- **Ingredient forecasting** - Link to recipe system
- **Vendor optimization** - Calculate Shamrock vs SYSCO savings
- **Margin analysis** - Compare 45% catering target vs actual

---

### 4. PaymentDetails.csv

**Contains:** Payment transaction details

**Key Fields:**

| Field Name | Type | Description |
|------------|------|-------------|
| **Payment Id** | Long | Unique payment identifier |
| **Order Id** | Long | Associated order |
| **Check Id** | Long | Associated check |
| **Amount** | Currency | Payment amount |
| **Tip** | Currency | Tip amount |
| **Gratuity** | Currency | Auto-gratuity |
| **Total** | Currency | Total payment |
| **Type** | String | Credit, Cash, Gift Card, etc. |
| **Card Type** | String | Visa, Mastercard, Amex, etc. |
| **Last 4 Card Digits** | String | Last 4 digits |
| **Status** | String | Approved, Declined, etc. |
| **V/MC/D Fees** | Currency | Processing fees |
| **Swiped Card Amount** | Currency | Swiped transaction amount |
| **Keyed Card Amount** | Currency | Manually keyed amount |
| **Refunded** | Boolean | TRUE if refunded |
| **Refund Date** | DateTime | Refund timestamp |
| **Refund Amount** | Currency | Refund amount |
| **Refund Tip Amount** | Currency | Refunded tip |
| **Cash Drawer** | String | Cash drawer identifier |
| **Dining Area** | String | Dining area name |
| **Service type** | String | Dine-in, Takeout, Delivery, Catering |

**Example Row:**
```csv
Payment Id,Order Id,Amount,Tip,Total,Type,Card Type,Last 4 Card Digits,Status,Service type
234567890123456,987654321012345,45.50,9.00,54.50,Credit,VISA,1234,Approved,Catering
```

**Use Cases for The Lariat:**
- Payment method analysis
- Credit card processing fees tracking
- Tip analysis by server/shift
- Refund tracking and reasons

---

### 5. AllItemsReport.csv

**Contains:** Menu item performance summary

**Key Fields:**

| Field Name | Type | Description |
|------------|------|-------------|
| **Master ID** | Long | Item master identifier |
| **Item ID** | Long | Item instance identifier |
| **Parent ID** | Long | Parent menu group ID |
| **Menu Item** | String | Item name |
| **SKU** | String | Stock-keeping unit |
| **Item Qty** | Integer | Quantity sold (with voids) |
| **Item Qty (without voids)** | Integer | Net quantity sold |
| **% of total quantity** | Decimal (5 places) | % of all items sold |
| **Gross Amount** | Currency | Revenue before discounts |
| **Net Amount** | Currency | Revenue after discounts |
| **Discount Amount** | Currency | Total discounts |
| **Void Amount** | Currency | Value of voided items |
| **% of total orders** | Decimal | % of all orders containing item |
| **% qty by group** | Decimal | % within menu group |
| **% qty by menu** | Decimal | % within menu |
| **% qty by all** | Decimal | % of all items |
| **% net amount by group** | Decimal | Revenue % within group |
| **% net amount by menu** | Decimal | Revenue % within menu |
| **% net amount by all** | Decimal | Revenue % of all items |
| **Avg Price** | Decimal (5 places) | Average selling price |

**Example Row:**
```csv
Menu Item,SKU,Item Qty (without voids),Net Amount,% of total orders,% net amount by all,Avg Price
Green Chile Bowl,GCB-REG,143,1857.57,12.45000,8.75000,12.99000
BBQ Pork Sandwich,BBQ-P,87,1015.50,9.10000,4.78000,11.67241
```

**Use Cases for The Lariat:**
- **Top sellers identification** - Focus on high-performing items
- **Slow movers** - Candidates for removal or promotion
- **Pricing analysis** - Compare avg price vs menu price (discounts)
- **Recipe costing integration** - Calculate food cost % by item

---

### 6. TimeEntries.csv (Labor Data Export)

**Contains:** Employee shift and labor information

**Key Fields:**

| Field Name | Type | Description |
|------------|------|-------------|
| **Employee ID** | String | Employee identifier |
| **Employee Name** | String | Full name |
| **Job Title** | String | Position/role |
| **Clock In** | DateTime | Shift start |
| **Clock Out** | DateTime | Shift end |
| **Break Start** | DateTime | Break start time |
| **Break End** | DateTime | Break end time |
| **Hours Worked** | Decimal | Total hours |
| **Regular Hours** | Decimal | Non-overtime hours |
| **Overtime Hours** | Decimal | Overtime hours |
| **Hourly Rate** | Currency | Pay rate |
| **Total Pay** | Currency | Calculated pay |
| **Tips Declared** | Currency | Declared tips |
| **Location** | String | Restaurant location |

**Example Row:**
```csv
Employee ID,Employee Name,Job Title,Clock In,Clock Out,Hours Worked,Hourly Rate,Total Pay,Tips Declared
EMP001,Sarah Johnson,Server,11/15/2025 10:00:00,11/15/2025 18:00:00,8.00,15.00,120.00,87.50
```

**Use Cases for The Lariat:**
- Labor cost % tracking (target: ~25-30%)
- Overtime management
- Staffing optimization
- Tip distribution analysis

---

### 7. KitchenTimings.csv

**Contains:** Kitchen ticket timing and performance

**Key Fields:**

| Field Name | Type | Description |
|------------|------|-------------|
| **Ticket ID** | String | Kitchen ticket identifier |
| **Order ID** | Long | Associated order |
| **Sent to Kitchen** | DateTime | Ticket sent time |
| **Prep Started** | DateTime | Prep began |
| **Ready for Pickup** | DateTime | Item completed |
| **Picked Up** | DateTime | Item delivered |
| **Total Time** | Integer | Minutes from sent to pickup |
| **Prep Time** | Integer | Minutes prep |
| **Wait Time** | Integer | Minutes waiting |
| **Menu Item** | String | Item name |
| **Prep Station** | String | Kitchen station |

**Example Row:**
```csv
Ticket ID,Order ID,Sent to Kitchen,Ready for Pickup,Total Time,Menu Item,Prep Station
KT-12345,987654321012345,11/15/2025 12:35:00,11/15/2025 12:47:00,12,Green Chile Bowl,Hot Line
```

**Use Cases for The Lariat:**
- Kitchen efficiency tracking
- Ticket time goals (target: <15 minutes)
- Bottleneck identification
- Staffing needs analysis

---

## API Structure

### API Overview

**Base URL:** `https://api.toasttab.com`

**Authentication:** OAuth 2.0 (Client Credentials or Authorization Code)

**Versions:**
- Menus API v2 (legacy)
- **Menus API v3** (current, recommended)

**Key APIs:**
1. **Menus API** - Menu structure, items, modifiers
2. **Orders API** - Create/modify orders
3. **Checks API** - Check management
4. **Labor API** - Employee schedules
5. **Stock API** - Inventory levels

---

### Menus API v3

#### Endpoints

**GET /menus/v3/menus**
- Retrieve complete menu structure
- Returns hierarchical JSON
- Includes all items, groups, modifiers

**GET /menus/v3/metadata**
- Check if menu data is stale
- Returns last modified timestamp
- Use for caching/sync decisions

#### MenuItem JSON Schema

```json
{
  "guid": "abc123-def456-ghi789",
  "multiLocationId": "500000000012345678",
  "masterId": 12345,
  "name": "Green Chile Bowl",
  "description": "Slow-cooked pork in Hatch green chile sauce",
  "posName": "GRN CHILE BOWL",
  "kitchenName": "GREEN CHILE",
  "plu": "1001",
  "sku": "GCB-REG",
  "calories": 650,
  "guestCount": 1.0,
  "isDiscountable": true,
  "isDeferred": false,
  "weight": 14.5,
  "dimensionUnitOfMeasure": "IN",
  "height": 3.5,
  "length": 8.0,
  "width": 8.0,
  "posButtonColorLight": "#90c490",
  "posButtonColorDark": "#348734",
  "image": "https://cdn.toasttab.com/images/...",
  "images": [
    "https://cdn.toasttab.com/images/image1.jpg",
    "https://cdn.toasttab.com/images/image2.jpg"
  ],
  "portions": [
    {
      "guid": "portion-guid-123",
      "name": "Regular",
      "price": 12.99,
      "isDefault": true
    },
    {
      "guid": "portion-guid-456",
      "name": "Large",
      "price": 15.99,
      "isDefault": false
    }
  ],
  "modifierGroupReferences": [
    700000000012345678,
    700000000012345679
  ],
  "itemTags": [
    {
      "guid": "tag-guid-001",
      "name": "Gluten-Free Available"
    }
  ],
  "allergens": [
    {
      "allergen": "DAIRY",
      "present": true
    },
    {
      "allergen": "GLUTEN",
      "present": false
    }
  ],
  "contentAdvisories": {
    "alcoholic": false,
    "vegetarian": false,
    "vegan": false,
    "raw": false,
    "spicy": "MEDIUM"
  },
  "eligiblePaymentAssistancePrograms": ["EBT", "WIC"]
}
```

**Key Fields Explained:**

- **guid** - Unique identifier for this specific item version
- **multiLocationId** - Identifier for versions across locations (preferred)
- **masterId** - Legacy identifier (deprecated, use multiLocationId)
- **name** - Display name on menu boards and online
- **posName** - Name shown on Toast POS (can be shorter)
- **kitchenName** - Name shown on kitchen tickets (even shorter)
- **plu** - Price Look-Up code for external integrations
- **sku** - Stock-Keeping Unit for inventory tracking
- **portions** - Multiple sizes/prices for same item
- **modifierGroupReferences** - Links to ModifierGroup objects
- **itemTags** - Searchable tags (dietary, cuisine type, etc.)
- **allergens** - Allergen information for safety
- **contentAdvisories** - Dietary classifications

#### ModifierGroup JSON Schema

```json
{
  "guid": "modifier-group-guid-123",
  "multiLocationId": "700000000012345678",
  "name": "Protein Options",
  "pricingMode": "PRICED_BY_MODIFIERS",
  "minSelections": 0,
  "maxSelections": 1,
  "defaultModifierGuids": [],
  "modifierOptions": [
    {
      "guid": "modifier-guid-001",
      "multiLocationId": "800000000012345678",
      "name": "Chicken",
      "price": 3.00,
      "isDefault": false,
      "displayMode": "DEFAULT"
    },
    {
      "guid": "modifier-guid-002",
      "multiLocationId": "800000000012345679",
      "name": "Pork",
      "price": 4.00,
      "isDefault": true,
      "displayMode": "DEFAULT"
    }
  ]
}
```

**ModifierGroup Fields:**

- **pricingMode** - `PRICED_BY_MODIFIERS` or `SIZE_PRICES`
- **minSelections** - Minimum required (0 = optional)
- **maxSelections** - Maximum allowed (1 = single choice, 999 = unlimited)
- **defaultModifierGuids** - Pre-selected modifiers
- **modifierOptions** - Array of available modifiers

#### MenuGroup JSON Schema

```json
{
  "guid": "menu-group-guid-123",
  "multiLocationId": "500000000032822323",
  "name": "Entrees",
  "description": "Our signature main dishes",
  "image": "https://cdn.toasttab.com/images/entrees.jpg",
  "menuItems": [
    {
      "guid": "menu-item-guid-001",
      "name": "Green Chile Bowl",
      "price": 12.99
      // ... (full MenuItem object)
    },
    {
      "guid": "menu-item-guid-002",
      "name": "BBQ Pork Sandwich",
      "price": 10.50
      // ... (full MenuItem object)
    }
  ],
  "subgroups": []
}
```

---

### API Authentication

**OAuth 2.0 Flow:**

1. **Register your application** at Toast Developer Portal
2. **Obtain credentials**: Client ID and Client Secret
3. **Request access token:**

```bash
POST https://api.toasttab.com/authentication/v1/authentication/login
Content-Type: application/json

{
  "clientId": "your-client-id",
  "clientSecret": "your-client-secret",
  "userAccessType": "TOAST_MACHINE_CLIENT"
}
```

4. **Response:**

```json
{
  "token": {
    "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI...",
    "expiresIn": 3600
  }
}
```

5. **Use token in requests:**

```bash
GET https://api.toasttab.com/menus/v3/menus
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI...
Toast-Restaurant-External-ID: your-restaurant-id
```

---

## Integration with The Lariat

### Data Flow Diagram

```
┌─────────────┐
│  Toast POS  │
│   (Live)    │
└──────┬──────┘
       │
       ├─ Nightly Exports (4 AM)
       │  ├─ OrderDetails.csv
       │  ├─ ItemSelectionDetails.csv
       │  ├─ PaymentDetails.csv
       │  ├─ TimeEntries.csv
       │  └─ AllItemsReport.csv
       │
       ├─ Menu Updates (as needed)
       │  └─ Bulk Import CSV
       │
       ▼
┌──────────────────────┐
│  The Lariat Bible    │
│  (Your System)       │
└──────┬───────────────┘
       │
       ├─ Import → modules/importers/file_importer.py
       │           ├─ import_toast_orders()
       │           ├─ import_toast_menu_items()
       │           └─ import_toast_sales()
       │
       ├─ Process → modules/recipes/recipe.py
       │            └─ Link menu items to recipes
       │
       ├─ Analyze → modules/vendor_analysis/
       │            └─ Calculate Shamrock vs SYSCO costs
       │
       └─ Report → Dashboard
                   ├─ Revenue: $28K catering + $20K restaurant
                   ├─ Margins: 45% catering, 4% restaurant
                   └─ Vendor savings: $52K/year
```

---

### Integration Use Cases

#### 1. Menu Item Sync

**Scenario:** Keep The Lariat's recipe system in sync with Toast POS menu

**Workflow:**
1. Export menu from Toast: `AllItemsReport.csv`
2. Import to Lariat: `file_importer.import_toast_menu_items()`
3. Match to recipes: Link by SKU or name
4. Calculate food cost: Recipe ingredients → Vendor prices
5. Analyze margins: Compare to 45% catering / 4% restaurant targets

**Implementation:**

```python
from modules.importers.file_importer import FileImporter

# Import Toast menu items
importer = FileImporter()
result = importer.import_toast_menu_items('data/toast/AllItemsReport.csv')

# Link to recipes
for item in result['items']:
    recipe = find_recipe_by_sku(item['sku'])
    if recipe:
        item['food_cost'] = recipe.calculate_cost()
        item['margin'] = (item['price'] - item['food_cost']) / item['price']

        # Flag items below target margin
        target = 0.45 if item['category'] == 'Catering' else 0.04
        if item['margin'] < target:
            print(f"⚠️  {item['name']}: {item['margin']:.1%} (target: {target:.1%})")
```

---

#### 2. Sales Analysis

**Scenario:** Track daily sales against revenue targets

**Workflow:**
1. Export orders: `OrderDetails.csv`
2. Import to Lariat: `file_importer.import_toast_orders()`
3. Categorize: Catering vs Restaurant
4. Calculate: Daily, weekly, monthly totals
5. Alert: If below pace for $28K catering / $20K restaurant

**Implementation:**

```python
# Import Toast orders
result = importer.import_toast_orders('data/toast/OrderDetails.csv')

# Analyze by revenue center
catering_sales = sum(o['total'] for o in result['orders'] if o['revenue_center'] == 'Catering')
restaurant_sales = sum(o['total'] for o in result['orders'] if o['revenue_center'] == 'Restaurant')

# Check against targets
print(f"Catering: ${catering_sales:,.2f} / $28,000 ({catering_sales/28000:.1%})")
print(f"Restaurant: ${restaurant_sales:,.2f} / $20,000 ({restaurant_sales/20000:.1%})")
```

---

#### 3. Vendor Cost Optimization

**Scenario:** Calculate savings from switching to Shamrock Foods

**Workflow:**
1. Export item sales: `ItemSelectionDetails.csv`
2. Match items to recipes
3. Look up ingredient costs: Shamrock vs SYSCO
4. Calculate potential savings
5. Generate recommendation report

**Implementation:**

```python
# Import Toast item sales
result = importer.import_toast_item_sales('data/toast/ItemSelectionDetails.csv')

total_savings = 0

for item in result['items']:
    recipe = find_recipe_by_name(item['menu_item'])
    if recipe:
        sysco_cost = recipe.calculate_cost(vendor='SYSCO')
        shamrock_cost = recipe.calculate_cost(vendor='Shamrock Foods')

        savings_per_item = sysco_cost - shamrock_cost
        total_savings += savings_per_item * item['qty']

# Annualize
annual_savings = total_savings * 365 / len(result['items'])
print(f"Estimated annual savings: ${annual_savings:,.2f}")
# Expected output: ~$52,000
```

---

#### 4. Labor Cost Management

**Scenario:** Track labor cost percentage

**Workflow:**
1. Export labor data: `TimeEntries.csv`
2. Import hours and pay
3. Calculate total labor cost
4. Divide by sales from OrderDetails.csv
5. Target: 25-30% labor cost

**Implementation:**

```python
# Import labor data
labor_result = importer.import_toast_labor('data/toast/TimeEntries.csv')
total_labor_cost = sum(e['total_pay'] + e['tips_declared'] for e in labor_result['entries'])

# Import sales
sales_result = importer.import_toast_orders('data/toast/OrderDetails.csv')
total_sales = sum(o['total'] for o in sales_result['orders'])

# Calculate labor %
labor_percentage = total_labor_cost / total_sales
print(f"Labor Cost %: {labor_percentage:.1%}")

if labor_percentage > 0.30:
    print("⚠️  Labor cost above 30% target")
```

---

### Recommended Import Schedule

**Daily (Automated):**
- `OrderDetails.csv` - Track daily sales
- `ItemSelectionDetails.csv` - Update item performance
- `PaymentDetails.csv` - Monitor payment methods

**Weekly:**
- `AllItemsReport.csv` - Menu item analysis
- `TimeEntries.csv` - Labor cost review
- `KitchenTimings.csv` - Kitchen efficiency

**Monthly:**
- Full menu export from Toast
- Reconcile with recipe costs
- Update vendor price comparisons
- Generate margin analysis report

**As Needed:**
- Menu updates via bulk import
- Price changes
- New item additions

---

## Template Examples

### Example 1: Creating a New Menu Item (Basic Template)

**Scenario:** Add "The Lariat Burger" to menu

**CSV Content:**
```csv
Operation,Entity type,Operation ID,Name,Parent entity type,Parent version ID or operation ID,Pricing strategy or method,Price
CREATE,MENU_ITEM,1,The Lariat Burger,MENU_GROUP,500000000032822323,BASE,13.99
```

**Steps:**
1. Open Basic Template in Google Sheets
2. Fill in row with data above
3. File → Download → CSV
4. Toast Web → Menus → Bulk Management → Bulk Import Tool
5. Upload CSV
6. Review and confirm import

---

### Example 2: Updating Prices (Item Update Template)

**Scenario:** Adjust prices for inflation

**CSV Content:**
```csv
Operation,Entity type,Operation ID,Version ID or operation ID,Price
UPDATE,MENU_ITEM,1,600000000045678901,13.99
UPDATE,MENU_ITEM,2,600000000045678902,11.50
UPDATE,MENU_ITEM,3,600000000045678903,15.99
```

**Steps:**
1. Export current menu to get Version IDs
2. Open Item Update Template
3. Fill in rows with new prices
4. Upload CSV to Toast

---

### Example 3: Creating Menu Item with Modifiers (Advanced Template)

**Scenario:** Add "Build Your Own Bowl" with protein options

**CSV Content:**
```csv
Operation,Entity type,Operation ID,Name,Parent entity type,Parent version ID or operation ID,Pricing strategy or method,Price,SKU,Button color
CREATE,MENU_ITEM,1,Build Your Own Bowl,MENU_GROUP,500000000032822323,BASE,10.99,BYO-BOWL,GREEN_75
CREATE,MODIFIER_GROUP,2,Protein Choice,MENU_ITEM,1,PRICED_BY_MODIFIERS,,
CREATE,MODIFIER,3,Chicken,MODIFIER_GROUP,2,BASE,3.00
CREATE,MODIFIER,4,Pork,MODIFIER_GROUP,2,BASE,4.00
CREATE,MODIFIER,5,Beef,MODIFIER_GROUP,2,BASE,5.00
CREATE,MODIFIER,6,No Protein,MODIFIER_GROUP,2,BASE,0.00
CREATE,MODIFIER_GROUP,7,Toppings,MENU_ITEM,1,PRICED_BY_MODIFIERS,,
CREATE,MODIFIER,8,Cheese,MODIFIER_GROUP,7,BASE,1.00
CREATE,MODIFIER,9,Sour Cream,MODIFIER_GROUP,7,BASE,0.75
CREATE,MODIFIER,10,Guacamole,MODIFIER_GROUP,7,BASE,2.00
```

**Result:**
- Base item: $10.99
- Customer chooses protein: +$3-5
- Customer adds toppings: +$0.75-2 each
- Final price: $14.74 (with chicken, cheese, sour cream)

---

### Example 4: Location-Specific Pricing (Advanced Template)

**Scenario:** Different price for Fort Collins vs Denver location

**CSV Content:**
```csv
Operation,Entity type,Operation ID,Name,Parent entity type,Parent version ID or operation ID,Pricing strategy or method,Location-specific price target ID,Price
CREATE,MENU_ITEM,1,Green Chile Bowl,MENU_GROUP,500000000032822323,LOCATION_SPECIFIC,LOC-FORTCOLLINS,12.99
CREATE,MENU_ITEM,1,Green Chile Bowl,MENU_GROUP,500000000032822323,LOCATION_SPECIFIC,LOC-DENVER,14.99
```

---

## Best Practices

### CSV Import Best Practices

1. **Always use templates**
   - Don't create CSV files from scratch
   - Use Toast-provided Google Sheets templates
   - Export to CSV only after filling out

2. **Test with small batches first**
   - Import 5-10 items as test
   - Verify results in Toast Web
   - Then import full menu

3. **Use Operation IDs consistently**
   - Sequential numbering: 1, 2, 3...
   - Helps track import progress
   - Easier to debug failures

4. **Maintain a master spreadsheet**
   - Keep local copy of all menu data
   - Track Version IDs for updates
   - Document changes with dates

5. **Price format validation**
   - Always use numeric strings: "10.99"
   - Never include currency symbols: "$"
   - Support negative for discounts: "-1.00"

6. **Handle GUIDs carefully**
   - Export current menu to get GUIDs
   - Copy-paste GUIDs exactly (no manual typing)
   - GUIDs are case-sensitive

7. **Backup before imports**
   - Export current menu before bulk import
   - Keep backup CSV for rollback
   - Document what changed

---

### Data Export Best Practices

1. **Set up automated exports**
   - Enable nightly exports (4 AM)
   - Configure FTP or CLI for automation
   - Download to secure location

2. **Archive historical data**
   - Keep daily exports for 90 days
   - Monthly archives for 7 years (tax purposes)
   - Compress old files to save space

3. **Monitor export failures**
   - Check daily for missing exports
   - Set up alerts for export errors
   - Have backup manual download process

4. **Use consistent column ordering**
   - Don't reorder columns frequently
   - Breaks import scripts
   - If reordering, update scripts first

5. **Filter exports appropriately**
   - Don't export unnecessary columns
   - Reduces file size
   - Faster downloads and imports

---

### API Best Practices

1. **Cache menu data**
   - Use `/metadata` endpoint to check staleness
   - Cache locally for 1 hour
   - Only fetch when changed

2. **Respect rate limits**
   - Toast API: 100 requests/minute
   - Implement exponential backoff
   - Queue requests during peak times

3. **Handle authentication tokens**
   - Tokens expire after 1 hour
   - Refresh before expiration
   - Store securely (never in code)

4. **Error handling**
   - Retry failed requests (3 times max)
   - Log all API errors
   - Alert on persistent failures

5. **Use webhook subscriptions**
   - Subscribe to order created events
   - Real-time updates vs polling
   - Reduces API calls

---

### The Lariat-Specific Recommendations

1. **Menu Management**
   - Update Toast menu from recipe system
   - Keep SKUs in sync with recipe IDs
   - Use POS Name for short kitchen display

2. **Vendor Integration**
   - Link Toast items to recipes
   - Calculate food cost from Shamrock/SYSCO prices
   - Flag items with margin below target

3. **Sales Tracking**
   - Import `OrderDetails.csv` daily
   - Separate catering vs restaurant revenue
   - Track against $28K / $20K targets

4. **Labor Management**
   - Import `TimeEntries.csv` weekly
   - Calculate labor cost %
   - Target 25-30% of sales

5. **Menu Engineering**
   - Import `ItemSelectionDetails.csv` monthly
   - Identify top sellers (stars)
   - Evaluate slow movers (dogs)
   - Adjust menu based on data

6. **Inventory Planning**
   - Link item sales to recipe ingredients
   - Forecast ingredient needs
   - Optimize vendor orders (Shamrock bulk discounts)

---

## Additional Resources

### Official Documentation

- **Toast Platform Guide:** https://doc.toasttab.com/doc/platformguide/
- **Developer Guide:** https://doc.toasttab.com/doc/devguide/
- **API Reference:** https://doc.toasttab.com/openapi/
- **Toast Central (Support):** https://central.toasttab.com/

### Developer Tools

- **Toast Developer Portal:** https://developer.toasttab.com/
- **API Sandbox:** Request access from developer portal
- **Postman Collection:** Available on developer portal

### Community

- **Toast Community Forum:** https://community.toasttab.com/
- **Feature Requests:** Submit via Toast Central
- **Status Page:** https://status.toasttab.com/

### Training

- **Toast University:** Free online courses
- **Webinars:** Monthly product updates
- **1-on-1 Training:** Contact Toast support

---

## Glossary

- **BEO** - Banquet Event Order (catering document)
- **GUID** - Globally Unique Identifier (Toast's record ID)
- **multiLocationId** - Identifier shared across restaurant locations
- **PLU** - Price Look-Up code (for external systems)
- **SKU** - Stock-Keeping Unit (inventory tracking)
- **Version ID** - Specific instance of a menu entity (used for updates)
- **Operation ID** - Temporary ID used during CSV imports
- **Entity Type** - Menu component type (MENU_ITEM, MODIFIER_GROUP, MODIFIER)
- **Pricing Strategy** - How item is priced (BASE, LOCATION_SPECIFIC, PRICED_BY_MODIFIERS)
- **Daypart** - Time period (Breakfast, Lunch, Dinner)
- **Revenue Center** - Business segment (Restaurant, Catering, Bar)

---

## Change Log

**2025-11-18:** Initial documentation
- Researched Toast POS CSV templates
- Documented data export formats
- Extracted API schemas (Menus v3)
- Created integration guides for The Lariat
- Added template examples and best practices

---

**For questions or support, contact:**
- Toast Support: https://central.toasttab.com/
- The Lariat System Admin: [your contact info]
