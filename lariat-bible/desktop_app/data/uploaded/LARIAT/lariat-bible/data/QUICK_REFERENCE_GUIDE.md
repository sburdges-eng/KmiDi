# Catering & Event Management Database - Quick Reference Guide
## For The Lariat Restaurant Management System

---

## Quick Facts

- **Sources Analyzed:** 3 professional event management systems from GitHub
- **Tables Identified:** 50+
- **Unique Fields Extracted:** 200+
- **Data Categories:** 15 major functional areas
- **Recommended Core Tables for Lariat:** 12

---

## Core Data Categories at a Glance

### 1. CUSTOMERS (Customer Master Data)
**Key Fields:** ID, Name (First + Last), Email, Phone, Address, Age, Priority Status, Discount
**Sample Use:** Track customer preferences, loyalty tiers, contact history
**Lariat Specific:** Track corporate vs. individual customers, catering preferences

### 2. EVENTS / BEO (Banquet Event Orders)
**Key Fields:** Event ID, Name, Type, Date, Start/End Times, Guest Count, Total Cost, Approval Status
**Event Types:** Wedding, Bridal Shower, Engagement, Birthday, Aqeeqa, Graduation, Corporate, Family Gathering
**Sample Use:** Complete event lifecycle from booking to completion
**Lariat Specific:** Link to Shamrock/SYSCO vendor selection, margin tracking

### 3. MENUS & FOOD ITEMS
**Key Fields:** Menu ID, Name, Type (Desi/Continental), Price, Cost, Specialty
**Sub-Components:** Rice type, Bread type, Protein, Beverages, Desserts
**Sample Use:** Build catering packages, track components, cost analysis
**Lariat Specific:** Store Shamrock and SYSCO pricing side-by-side, track savings

### 4. VENUES
**Key Fields:** Venue ID, Name, Location, Address, Capacity, Category, Rental Cost
**Categories:** Indoor Banquet Hall, Outdoor, Riverside, Garden, Wedding Hall
**Sample Use:** Venue availability, capacity matching, cost estimation
**Lariat Specific:** Link venue capacity to menu portion planning

### 5. CATERING VENDORS
**Key Fields:** Vendor ID, Name, Specialty, Contact, Service Charges, Minimum Days
**Primary Vendors:** Shamrock Foods (PREFERRED), SYSCO (Comparison), Local options
**Sample Use:** Supplier management, order placement, quality tracking
**Lariat Specific:** THIS IS THE DIFFERENTIATOR - Shamrock is ~29.5% cheaper than SYSCO

### 6. STAFF & PERSONNEL
**Key Fields:** Employee ID, Name, DOB, Email, Phone, Department, Role, Salary/Wage, Manager
**Roles:** Chef, Sous Chef, Prep Cook, Server, Host, Event Manager, Decorator
**Sample Use:** Scheduling, payroll, performance tracking
**Lariat Specific:** Track staff assignments to specific events, manage kitchen capacity

### 7. ORDERS & LINE ITEMS
**Key Fields:** Order ID, Status, Subtotal, Tax, Total, Customer Reference, Items, Quantities
**Statuses:** Pending, Confirmed, Completed, Cancelled
**Sample Use:** Order management, fulfillment, billing
**Lariat Specific:** Calculate margins at order level, track vendor costs per order

### 8. FINANCIAL TRACKING
**Key Fields:** Subtotal, Tax, Total, Unit Prices, Service Charges, Discounts
**Calculations:**
  - Margin = (Menu Price - Food Cost) / Menu Price
  - Target Catering Margin: 45%
  - Target Restaurant Margin: 4%
**Sample Use:** Pricing optimization, profitability analysis
**Lariat Specific:** Track actual vs. target margins, identify underpriced items

### 9. VENDOR COMPARISON (Shamrock vs SYSCO)
**Key Fields:** Product Name, Shamrock Price, SYSCO Price, Savings %, Preferred Vendor
**Potential Annual Savings:** $52,000 (29.5% cheaper with Shamrock)
**Sample Use:** Vendor selection, procurement optimization
**Lariat Specific:** Core differentiator - automate vendor comparison for each menu item

### 10. PHOTOGRAPHY & MEDIA
**Key Fields:** Photographer ID, Name, Charges, Services (Photography, Videography, Drone, Album)
**Sample Use:** Media vendor management, event coverage planning
**Lariat Specific:** Link to event for complete service packaging

### 11. DECORATION & FLORAL
**Key Fields:** Decorator ID, Name, Service Type, Cost, Address
**Service Types:** Floral, Lighting, Backdrops, Centerpieces, Themed Decor
**Sample Use:** Decoration vendor management
**Lariat Specific:** Optional service for premium catering packages

### 12. TRANSPORTATION
**Key Fields:** Vehicle Type, Company, Cost per KM, Seating Capacity, Contact
**Sample Use:** Logistics planning, guest transportation
**Lariat Specific:** Additional service offering for destination events

### 13. PERFORMERS & ENTERTAINMENT
**Key Fields:** Performer ID, Name, Type (Dancer/Musician/Comedian), Specialty, Charges
**Sample Use:** Entertainment vendor management
**Lariat Specific:** Optional premium add-on service

### 14. EQUIPMENT & KITCHEN INVENTORY
**Key Fields:** Equipment ID, Name, Cost, Supplier, Maintenance Schedule
**Sample Use:** Asset tracking, maintenance scheduling
**Lariat Specific:** Critical for kitchen capacity planning, maintenance cost tracking

### 15. AUTHENTICATION & ACCOUNTS
**Key Fields:** Username, Password (Hashed), Account ID, User Type
**User Types:** Customer, Employee, Manager, Admin
**Sample Use:** User access control, security
**Lariat Specific:** Role-based access for staff, customers, vendors

---

## Field Name Conventions Used Across Systems

### ID/Primary Keys
| Field Purpose | Common Names |
|---------------|--------------|
| Customer | cust_id, customer_id, id |
| Event | event_id, eve_id, id |
| Menu | menu_id, id |
| Venue | venue_id, v_id, id |
| Employee | emp_id, id |
| Order | order_id, id |

### Date/Time Fields
| Field Purpose | Common Names | Format |
|---------------|--------------|--------|
| Creation Date | register_date, order_date, created | YYYY-MM-DD HH:MM:SS |
| Event Date | event_date, date | YYYY-MM-DD |
| Time | starting_time, ending_time, time | HH:MM or "5:00 PM" |
| Birth Date | dob, birthday | YYYY-MM-DD |

### Price Fields
| Field Purpose | Common Names | Type |
|---------------|--------------|------|
| Menu Price | price, menu_price, cost | DECIMAL(10,2) |
| Food Cost | food_cost, cost | DECIMAL(10,2) |
| Service Charge | charges, service_charge | NUMERIC(38,0) |
| Total | total, total_cost | BIGDECIMAL |
| Tax | tax | BIGDECIMAL |

---

## Critical Calculations for The Lariat

### Margin Calculation (Most Important)
```
Margin Percentage = (Menu Price - Food Cost) / Menu Price

Example:
Menu Price: $45.99 (Shamrock Steak)
Food Cost: $18.50 (from Shamrock)
Margin = (45.99 - 18.50) / 45.99 = 27.49 / 45.99 = 59.8%
Target Catering Margin: 45%
Variance: +14.8% (ABOVE TARGET)
Recommendation: Price could be optimized downward or this is excellent profitability
```

### Vendor Savings Calculation
```
Savings per Unit = SYSCO Price - Shamrock Price
Savings Percentage = (Shamrock Price - SYSCO Price) / SYSCO Price * 100

Example:
Black Pepper Fine (1LB)
SYSCO: $12.99
Shamrock: $8.99
Savings: $12.99 - $8.99 = $4.00
Savings %: (8.99 - 12.99) / 12.99 * 100 = 30.8%

Annual Impact (if 5,000 units/year):
$4.00 per unit × 5,000 = $20,000 ANNUAL SAVINGS
Multiply across all items = $52,000 TOTAL POTENTIAL SAVINGS
```

### Suggested Price at Target Margin
```
Suggested Price = Food Cost / (1 - Target Margin)

Example:
Food Cost: $18.50
Target Margin: 45% (0.45)
Suggested Price = 18.50 / (1 - 0.45)
                = 18.50 / 0.55
                = $33.64

This price would deliver exactly 45% margin
Current Price: $45.99
Current Margin: 59.8%
Opportunity: Could reduce price to be more competitive while maintaining 45% margin
```

---

## Relationship Overview (Entity Diagram)

```
CUSTOMER
  ├── Account (authentication)
  ├── Orders (multiple)
  │   └── OrderItems
  │       └── Menu (referenced)
  └── Events/BEOs (multiple)
      ├── Venue
      ├── Catering Vendor
      ├── Menu Selection
      ├── Staff Assignments
      ├── Photography/Media
      ├── Decorators
      ├── Equipment Needs
      ├── Transportation
      └── Performers

MENU
  ├── Catering Vendor
  └── Vendor Comparison (Shamrock vs SYSCO pricing)

STAFF
  ├── Manager (reporting)
  ├── Department Assignment
  └── Event Assignments (multiple events)

VENDOR (Catering, Decoration, Photography, Transport)
  └── Services/Products offered
```

---

## Data Import/Export Formats

### Recommended Formats for Lariat
1. **CSV** - For Excel integration, reporting, bulk import/export
2. **JSON** - For API integration, system-to-system communication
3. **SQL** - For database initialization, backups, migrations
4. **Excel** - For user-friendly data entry and reporting

### Sample File Locations
- Customers: `/data/customers.csv`
- Events: `/data/events.csv`
- Menus: `/data/menus.csv`
- Vendor Comparison: `/data/vendor_comparison.csv`
- Margins: `/data/margin_analysis.csv`

---

## Normalization Best Practices Observed

1. **Multi-valued Fields Split to Separate Tables**
   - Example: Customer phones in separate `customer_phones` table
   - Example: Employee phones in separate `employee_phones` table
   - Example: Venue phones in separate `venue_phones` table

2. **Specialization/ISA Relationships**
   - Employees → Managers, Logistics, Chefs (specialized tables)
   - Performers → Dancers, Musicians, Fillers, Anchors (specialized)

3. **Junction Tables for Many-to-Many**
   - `appointed_to`: Employees to Events
   - `performs_in`: Performers to Events
   - `served_by`: Caterers to Food Types
   - `needs`: Equipment to Events

4. **Candidate Keys Tracked**
   - Email as unique identifier
   - Username as unique identifier
   - Natural keys where possible (name + address combinations)

---

## Status/Enumeration Values

### Order Status
- 0 = Pending (not yet confirmed)
- 1 = Confirmed (accepted by customer)
- 2 = Completed (event has occurred)
- 3 = Cancelled (customer cancelled)

### Event Approval
- 0 = Pending Review
- 1 = Approved
- 2 = Rejected

### Employee Status
- 0 = Inactive
- 1 = Active
- 2 = On Leave
- 3 = Terminated

### Menu Status
- 0 = Inactive (not available)
- 1 = Active (available for booking)

### Priority Status (Customer)
- 1 = VIP
- 2 = Regular
- 3 = Standard

---

## Implementation Roadmap for The Lariat

### Phase 1 (MVP) - Core Tables
1. ✓ Customers
2. ✓ Events/BEOs
3. ✓ Menus
4. ✓ Catering Orders
5. ✓ Vendor Pricing (Shamrock vs SYSCO)

### Phase 2 - Extended Services
6. Venues
7. Staff Management
8. Order Tracking
9. Margin Analysis
10. Vendor Comparison Reports

### Phase 3 - Advanced Features
11. Equipment Inventory & Maintenance
12. Photography/Media Services
13. Decoration Services
14. Transportation Services
15. Performance Analytics

### Phase 4 - Optimization
16. Pricing Intelligence
17. Predictive Margins
18. Vendor Optimization
19. Capacity Planning
20. Advanced Reporting

---

## Key Performance Indicators (KPIs) to Track

### Financial KPIs
- **Average Margin %** - Target: 45% for catering, 4% for restaurant
- **Vendor Savings %** - Target: Maintain 29.5% savings with Shamrock
- **Monthly Revenue** - Catering: $28K, Restaurant: $20K
- **Food Cost %** - Lower is better, track by vendor

### Operational KPIs
- **Order Fulfillment Rate** - % of orders completed on schedule
- **Customer Satisfaction** - Post-event feedback scores
- **Staff Utilization** - % of assigned events per staff member
- **Vendor Performance** - On-time delivery, quality ratings

### Strategic KPIs
- **Catering vs Restaurant Mix** - % of revenue from each
- **Repeat Customer Rate** - % of returning customers
- **Average Order Value** - Trend analysis over time
- **Shamrock Utilization %** - % of orders using preferred vendor

---

## Common Queries You'll Want

### Find All Events for a Customer
```sql
SELECT e.* FROM events e
WHERE e.cust_id = 'C001'
ORDER BY e.event_date DESC
```

### Calculate Margin on All Orders
```sql
SELECT
  o.order_id,
  o.total as menu_price,
  SUM(oi.cost) as food_cost,
  (o.total - SUM(oi.cost)) / o.total as margin_percentage
FROM orders o
JOIN order_items oi ON o.id = oi.order_id
GROUP BY o.order_id
```

### Compare Shamrock vs SYSCO Savings
```sql
SELECT
  product_name,
  shamrock_price,
  sysco_price,
  (sysco_price - shamrock_price) as savings_per_unit,
  ROUND((sysco_price - shamrock_price) / sysco_price * 100, 1) as savings_percentage
FROM vendor_comparison
WHERE shamrock_price < sysco_price
ORDER BY savings_percentage DESC
```

### Find Events Above Margin Target
```sql
SELECT
  e.event_id, e.name, e.total_cost,
  SUM(oi.cost) as food_cost,
  (e.total_cost - SUM(oi.cost)) / e.total_cost as actual_margin,
  0.45 as target_margin
FROM events e
JOIN order_items oi ON e.order_id = oi.order_id
GROUP BY e.event_id
HAVING (e.total_cost - SUM(oi.cost)) / e.total_cost > 0.45
```

---

## File References in This Package

1. **CATERING_EVENT_MANAGEMENT_SCHEMA.md** - Comprehensive schema documentation
2. **UNIFIED_FIELD_CATALOG.csv** - All fields in spreadsheet format
3. **SAMPLE_DATA_TEMPLATES.csv** - Sample data rows for each table
4. **SAMPLE_DATA_TEMPLATES.json** - Sample data in JSON format
5. **QUICK_REFERENCE_GUIDE.md** - This file

---

## Next Steps for Lariat Bible Implementation

1. **Review** the complete schema in `CATERING_EVENT_MANAGEMENT_SCHEMA.md`
2. **Map** existing Lariat data to the recommended tables
3. **Extend** with custom Lariat fields:
   - Shamrock vs SYSCO vendor flags
   - Margin analysis calculations
   - Equipment maintenance tracking
   - Seasonal pricing adjustments
4. **Implement** margin tracking at order level
5. **Configure** reporting for vendor comparison
6. **Train** staff on the new system structure
7. **Validate** data accuracy before going live

---

## Critical Business Rules for The Lariat

1. **Always use Shamrock Foods by default** (29.5% cost advantage)
2. **Verify product specifications match** (pepper grind, case size, quality)
3. **Track margins at order level** to identify pricing problems
4. **Flag items below 45% catering margin** for review
5. **Maintain separate pricing** for different event types
6. **Monitor customer types** (corporate vs. individual, repeat vs. new)
7. **Track equipment availability** for capacity planning
8. **Review staffing efficiency** monthly
9. **Analyze vendor performance** quarterly
10. **Benchmark pricing** against competitors semi-annually

---

## Support Resources

- **CLAUDE.md** - AI assistant guide for this codebase
- **GitHub References** - See source repositories for implementation examples
- **Sample Templates** - Use provided CSV/JSON samples for testing
- **Normalization Examples** - See schema document for database design patterns

---

**Document Version:** 1.0
**Last Updated:** November 18, 2025
**Maintained By:** Lariat Bible System
**Contact:** Sean (Restaurant Owner)

---

## Quick Checklist for Lariat System Setup

- [ ] Database created with identified core tables
- [ ] Customer master data imported
- [ ] Event/BEO templates configured
- [ ] Menu items entered with Shamrock & SYSCO pricing
- [ ] Staff records loaded
- [ ] Venue master list created
- [ ] Vendor contact information documented
- [ ] Margin calculation logic implemented
- [ ] Vendor comparison reports configured
- [ ] Staff training completed
- [ ] Test events processed successfully
- [ ] Go-live approved by Sean

