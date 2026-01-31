# Comprehensive Catering & Event Management Database Schema
## Extracted from GitHub Reference Implementations

**Analysis Date:** November 18, 2025
**Repositories Analyzed:**
1. https://github.com/jeel-shah24/Event-management-system
2. https://github.com/HxnDev/Event-Management-System
3. https://github.com/ling67/Web-Application-Catering-Management-System

---

## Executive Summary

This document consolidates database schemas from three professional event and catering management systems. It provides a comprehensive list of all unique fields used for BEO (Banquet Event Order) management, catering events, and event logistics. The schemas are organized by functional category to support Lariat Bible's catering operations.

---

## SECTION 1: EVENT CORE DETAILS

### Primary Tables: `event`, `event_detail`

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| event_id / id | VARCHAR(20) / INTEGER | Unique event identifier | JEel-Shah, HxnDev, Ling67 |
| event_name / name | VARCHAR(30-50) | Name/title of event | All Three |
| event_type / type | VARCHAR(25-30) | Event category (Wedding, Bridal Shower, Engagement, Birthday, Aqeeqa, Graduation, Valima, Baraat, Bachelors Party, Thanks Giving, Nikkah) | All Three |
| event_date | DATE | Date event is scheduled | All Three |
| starting_time | VARCHAR(30) / TIME | Event start time | JEel-Shah, HxnDev |
| ending_time | VARCHAR(30) / TIME | Event end time | JEel-Shah, HxnDev |
| eve_feedback / feedback | VARCHAR(59-200) | Post-event feedback/notes | Jeel-Shah, HxnDev |
| guests / max_guests | NUMERIC / INTEGER | Expected number of guests | HxnDev, Ling67 |
| approved | TINYINT (0/1) | Event approval status | HxnDev |
| total_cost | NUMERIC / BIGINT | Final total event cost | HxnDev, Ling67 |
| order_date / created | DATE / DATETIME | Order/creation date timestamp | Ling67, Web-Catering |
| status / order_status | INTEGER / SMALLINT | Order status (pending, confirmed, completed) | Web-Catering |

### Order-Specific Fields (Web-Catering System)

| Field Name | Data Type | Description |
|-----------|-----------|-------------|
| subtotal | BIGDECIMAL | Pre-tax order amount |
| tax | BIGDECIMAL | Tax amount |
| total | BIGDECIMAL | Total after tax |

---

## SECTION 2: CUSTOMER & CLIENT INFORMATION

### Primary Tables: `customer`, `customer1`, `customer2`

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| cust_id / customer_id / id | VARCHAR(20) / INTEGER | Unique customer identifier | All Three |
| first_name / c_fname | VARCHAR(20-30) | Customer first name | Jeel-Shah, HxnDev |
| last_name / c_lname | VARCHAR(25-30) | Customer last name | Jeel-Shah, HxnDev |
| name | VARCHAR(30) | Full customer name | HxnDev |
| email / email_id / email | VARCHAR(30-45) | Customer email address | All Three |
| phone_no / c_phone / phone_number | VARCHAR(11-30) | Customer phone number | All Three |
| address / c_address | VARCHAR(45-100) | Customer mailing address | All Three |
| cnic | VARCHAR(20) | National ID number | HxnDev |
| age | DOUBLE | Customer age | HxnDev |
| account_number | VARCHAR(20) | Banking/payment account number | HxnDev |
| priority_status | NUMERIC(1-3) | Customer priority level (VIP tier) | HxnDev |
| username | VARCHAR(45) | Customer login username | Jeel-Shah |
| password | VARCHAR(45) | Customer login password (encrypted) | Jeel-Shah, HxnDev |
| discount | FLOAT | Customer discount percentage | Jeel-Shah |
| register_date | DATETIME | Account registration timestamp | Web-Catering |

**Note:** Customer Phone and Password often stored in separate related tables for normalization.

---

## SECTION 3: MENU, FOOD & CATERING SERVICES

### Primary Tables: `menu`, `food`, `catering`

#### Menu/Catering Services Table

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| menu_id / catering_id | VARCHAR(5-20) / INTEGER | Menu/catering service identifier | All Three |
| menu_name / name | VARCHAR(20-100) | Descriptive catering/food name | All Three |
| menu_type / specialty | VARCHAR(30-100) | Type (Desi, Continental, Vegetarian, etc.) or Buffet style | HxnDev, Ling67 |
| f_type / food_type | VARCHAR(45) | Specific food category (Rice, Bread, Protein, Dessert) | Jeel-Shah |
| cost | DECIMAL(10-20) / NUMERIC(38,0) | Unit price of menu item | All Three |
| price | BIGDECIMAL | Menu item price | Web-Catering |
| charges | NUMERIC(38,0) | Service charges for catering (if separate from cost) | HxnDev |
| days | NUMERIC(38,0) | Available service days (min days booking) | HxnDev |
| description | VARCHAR(200) | Menu item description/details | Web-Catering |
| menuImage | VARCHAR | Image path/URL for menu visualization | Web-Catering |
| status / menu_status | INTEGER / TINYINT | Active/inactive status of menu item | Web-Catering, HxnDev |
| contact / contact_info | VARCHAR(30) | Catering company contact | HxnDev |

#### Detailed Food Components

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| rice | VARCHAR(100) | Rice dish options (Biryani, Plain Boiled, Egg-fried, Kabuli) | HxnDev |
| bread | VARCHAR(100) | Bread options (Naan, Tandoori, Lebanese) | HxnDev |
| protein | VARCHAR(100) | Protein options (Chicken, Beef, Seafood) | HxnDev |
| coke / miranda / sprite / water | TINYINT (0/1) | Individual beverage inclusion flags | HxnDev |
| dryfruit | TINYINT (0/1) | Dry fruit dessert inclusion | HxnDev |
| sugarfree | TINYINT (0/1) | Sugar-free option inclusion | HxnDev |
| icecream | TINYINT (0/1) | Ice cream inclusion | HxnDev |
| cake | TINYINT (0/1) | Cake inclusion | HxnDev |

---

## SECTION 4: VENUE & LOCATION DETAILS

### Primary Tables: `venue`, `venue1`, `venue2`, `venueaddress1`, `venueaddress2`

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| venue_id / v_id | VARCHAR(20) / CHAR(5) | Unique venue identifier | All Three |
| venue_name / v_name | VARCHAR(20-45) | Venue name | All Three |
| location | VARCHAR(50) | Geographic location/city area | HxnDev |
| address / v_add | VARCHAR(45-100) | Full venue address | All Three |
| capacity / max_capacity | BIGINT(20) / NUMERIC | Maximum guest capacity | All Three |
| v_rent | FLOAT / NUMERIC | Venue rental cost | Jeel-Shah, HxnDev |
| category | VARCHAR(30) | Venue type (Indoor, Outdoor, Riverside, Banquet Hall, Wedding Garden) | HxnDev |
| description | VARCHAR(200) | Venue features/amenities description | HxnDev |
| contact_info / v_phn | VARCHAR(30) | Venue contact phone number | All Three |
| v_pincode | VARCHAR(8) | Postal code | Jeel-Shah |
| city | VARCHAR(15) | City name | Jeel-Shah |
| state | VARCHAR(15) | State/province | Jeel-Shah |
| line1 / line2 | VARCHAR(45) | Address line components | Jeel-Shah |

---

## SECTION 5: LOGISTICS, TRANSPORTATION & SERVICES

### Primary Tables: `transportation`, `provided`, `studio`, `media_requirements`

#### Transportation Services

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| vehicle_name / v_name | VARCHAR(40) | Vehicle/transportation type | All Three |
| cost_per_km | FLOAT | Cost per kilometer | Jeel-Shah |
| no_of_seats | INT | Vehicle seating capacity | Jeel-Shah |
| studio_id | CHAR(5) | Photography/videography studio identifier | HxnDev |
| t_id | VARCHAR(20) | Transportation company ID | Jeel-Shah, HxnDev |
| t_name | VARCHAR(25) | Transportation company name | Jeel-Shah, HxnDev |
| t_add | VARCHAR(45) | Company address | Jeel-Shah |
| t_phn | BIGINT(11) | Company phone | Jeel-Shah |

#### Media & Photography Requirements

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| media_id | CHAR(5) | Media requirements identifier | HxnDev |
| photography | TINYINT (0/1) | Photography service required | HxnDev |
| videography | TINYINT (0/1) | Videography service required | HxnDev |
| album | TINYINT (0/1) | Album/prints required | HxnDev |
| drone | TINYINT (0/1) | Drone coverage required | HxnDev |
| crane | TINYINT (0/1) | Crane/elevated shots required | HxnDev |

#### Photography Studio Services

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| studio_id | CHAR(5) | Studio identifier | HxnDev |
| studio_name | VARCHAR(30) | Photography studio name | HxnDev |
| contact_info | VARCHAR(30) | Studio contact | HxnDev |
| cost | NUMERIC(38,0) | Studio service cost | HxnDev |

#### Photography Personnel

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| p_id / photographer_id | VARCHAR(20) | Photographer identifier | Jeel-Shah |
| p_fname | VARCHAR(20) | Photographer first name | Jeel-Shah |
| p_lname | VARCHAR(20) | Photographer last name | Jeel-Shah |
| p_add | VARCHAR(40) | Photographer address | Jeel-Shah |
| p_mail_id | VARCHAR(20) | Photographer email | Jeel-Shah |
| p_charges | DECIMAL(20) | Photographer charges | Jeel-Shah |

---

## SECTION 6: FINANCIAL & COST MANAGEMENT

### Cost-Related Fields (Distributed Across Tables)

| Field Name | Data Type | Purpose | Source Systems |
|-----------|-----------|---------|-----------------|
| subtotal | BIGDECIMAL | Pre-tax order amount | Web-Catering |
| tax | BIGDECIMAL | Tax calculation on order | Web-Catering |
| total | BIGDECIMAL | Final order total | Web-Catering |
| total_cost | NUMERIC / BIGINT | Event total cost | HxnDev, Ling67 |
| menu_cost | NUMERIC(38,0) | Menu item cost | HxnDev |
| catering_charges | NUMERIC(38,0) | Catering service charge | HxnDev |
| venue_rent | FLOAT | Venue rental cost | Jeel-Shah |
| v_rent | FLOAT | Venue rental (alt naming) | Jeel-Shah |
| cost_per_km | FLOAT | Transportation cost per km | Jeel-Shah |
| p_charges | DECIMAL(20) | Photography charges | Jeel-Shah |
| studio_cost | NUMERIC(38,0) | Studio service cost | HxnDev |
| discount | FLOAT | Customer discount (%) | Jeel-Shah |
| eve_trans_charges | DECIMAL(10) | Transportation charges specific to event | Jeel-Shah |

**Margin Calculations (Lariat-Specific):**
- Catering Target Margin: 45% (0.45)
- Restaurant Target Margin: 4% (0.04)
- Formula: `margin = (menu_price - food_cost) / menu_price`
- Suggested Price: `food_cost / (1 - target_margin)`

---

## SECTION 7: PERSONNEL & STAFF MANAGEMENT

### Primary Tables: `employee`, `employees1`, `employees2`, `manager`, `logistics`, `performers`

#### Employee/Staff Core Information

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| emp_id | VARCHAR(20) | Employee identifier | All Three |
| emp_fname / name | VARCHAR(20-30) | Employee first name | All Three |
| emp_lname | VARCHAR(20) | Employee last name | All Three |
| emp_add | VARCHAR(45) | Employee address | All Three |
| emp_phn / phone_no | VARCHAR(11-30) | Employee contact phone | All Three |
| emp_salary | DECIMAL(10) | Employee salary | Jeel-Shah |
| wage_type | VARCHAR(10) | Wage payment method (Hourly, Daily, Monthly) | HxnDev |
| wage_rate | NUMERIC(10) | Hourly/daily/monthly rate | HxnDev |
| dob | DATE | Date of birth | HxnDev |
| email | VARCHAR(30) | Employee email | HxnDev |
| cnic | VARCHAR(20) | National ID | HxnDev |
| account_number | VARCHAR(20) | Banking account for wage payment | HxnDev |
| points | NUMERIC(10) | Performance/loyalty points | HxnDev |
| password | VARCHAR(30) | Login password | HxnDev |
| mgr_id | VARCHAR(20) | Manager ID (reporting relationship) | HxnDev |
| register_date | DATETIME | Employee registration/hire date | Web-Catering |

#### Manager-Specific Fields

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| emp_id (FK) | VARCHAR(20) | Reference to employee | Jeel-Shah |
| department | VARCHAR(25) | Department assignment | Jeel-Shah |

#### Logistics Team Fields

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| emp_id (FK) | VARCHAR(20) | Reference to employee | Jeel-Shah |
| workallotted | VARCHAR(30) | Work assignment/task | Jeel-Shah |

#### Performer/Entertainer Fields

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| per_id | VARCHAR(20) | Performer identifier | Jeel-Shah |
| per_fname | VARCHAR(20) | Performer first name | Jeel-Shah |
| per_lname | VARCHAR(20) | Performer last name | Jeel-Shah |
| per_mail_id | VARCHAR(25) | Performer email | Jeel-Shah |
| per_add | VARCHAR(45) | Performer address | Jeel-Shah |
| per_salary | Salary payment amount | Jeel-Shah |

#### Performer Specialization

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| per_id (FK) | VARCHAR(20) | Reference to performer | Jeel-Shah |
| forte | VARCHAR(40) | Dancer specialty/style | Jeel-Shah |
| mus_type | VARCHAR(30) | Musician type/instrument | Jeel-Shah |
| fil_type | VARCHAR(30) | Filler type (comedian, mime, etc.) | Jeel-Shah |
| gender | CHAR(1) | Anchor gender | Jeel-Shah |

#### Makeup & Beauty Services

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| m_type / makeup_type | VARCHAR(45) | Makeup type/style | Jeel-Shah |
| m_charge | DECIMAL(10) | Makeup service charge | Jeel-Shah |
| par_id | VARCHAR(20) | Parlour/salon identifier | Jeel-Shah |
| par_name | VARCHAR(45) | Parlour/salon name | Jeel-Shah |
| par_mail_id | VARCHAR(35) | Parlour email | Jeel-Shah |
| par_add | VARCHAR(50) | Parlour address | Jeel-Shah |

#### Decoration Services

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| dec_id | VARCHAR(20) | Decorator identifier | Jeel-Shah |
| dec_add | VARCHAR(45) | Decorator address | Jeel-Shah |
| dec_mail_id | VARCHAR(20) | Decorator email | Jeel-Shah |
| dec_name | VARCHAR(45) | Decorator name | Jeel-Shah |
| d_type / decoration_type | VARCHAR(30) | Decoration type (Floral, Lighting, etc.) | Jeel-Shah |
| d_charge / charge | DECIMAL(10) | Decoration service charge | Jeel-Shah |

#### Electronics/Equipment Supplier

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| e_id | VARCHAR(20) | Equipment company identifier | Jeel-Shah |
| e_add | VARCHAR(45) | Company address | Jeel-Shah |
| e_name | VARCHAR(45) | Company name | Jeel-Shah |
| e_mail_id | VARCHAR(20) | Company email | Jeel-Shah |
| el_id | VARCHAR(20) | Equipment identifier | Jeel-Shah |
| el_name | VARCHAR(45) | Equipment name | Jeel-Shah |
| cost | DECIMAL(20) | Equipment cost | Jeel-Shah |

#### Caterer Information

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| cat_id | VARCHAR(20) | Caterer identifier | Jeel-Shah, HxnDev |
| cat_name | VARCHAR(45) | Caterer business name | Jeel-Shah, HxnDev |
| cat_mail_id | VARCHAR(20) | Caterer email | Jeel-Shah, HxnDev |
| cat_add | VARCHAR(45) | Caterer address | Jeel-Shah, HxnDev |
| contact | VARCHAR(30) | Contact phone | HxnDev |
| specialty | VARCHAR(30) | Specialty cuisine (Desi, Continental) | HxnDev |
| days | NUMERIC(38,0) | Minimum booking days | HxnDev |
| charges | NUMERIC(38,0) | Catering charges | HxnDev |

---

## SECTION 8: ORDER & TRANSACTION MANAGEMENT

### Primary Tables: `order_detail`, `order_item`, `orderItem`

#### Order Header

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| id / order_id | INTEGER / VARCHAR(20) | Unique order identifier | Web-Catering, Jeel-Shah |
| status / order_status | INTEGER / SMALLINT | Order status code | All Three |
| subtotal | BIGDECIMAL | Pre-tax line items total | Web-Catering |
| tax | BIGDECIMAL | Tax amount | Web-Catering |
| total | BIGDECIMAL | Final order total | Web-Catering |
| created / order_date | DATETIME / DATE | Order creation timestamp | Web-Catering, HxnDev |
| customer_id (FK) | INTEGER / VARCHAR(20) | Customer reference | All Three |

#### Order Line Items

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| itemId / id | INTEGER | Order item identifier | Web-Catering |
| order_id (FK) | INTEGER | Reference to parent order | Web-Catering |
| menu_id (FK) | INTEGER / VARCHAR(20) | Reference to menu item | Web-Catering, Ling67 |
| item_quantity | INTEGER | Quantity ordered | Web-Catering |

---

## SECTION 9: RELATIONAL INTEGRITY & ASSOCIATIONS

### Junction Tables for Many-to-Many Relationships

| Relationship | Primary Keys | Foreign Keys | Purpose |
|-------------|--------------|--------------|---------|
| `appointed_to` | (emp_id, eve_id) | employees → event | Assigns employees to events |
| `conducted` | (I_id, eve_id) | informals → event | Links informal services to events |
| `supplied_by` | (e_id, el_id) | ecompany → equipment | Equipment supplier relationships |
| `provided` | (t_id, v_name) | t_company → transportation | Transportation provider services |
| `served_by` | (cat_id, f_type) | caterers → food | Caterer food specialties |
| `arranged_for` | (v_name, eve_id) | transportation → event | Transportation for specific events |
| `needs` | (el_id, eve_id) | equipment → event | Equipment needed for event |
| `performs_in` | (per_id, eve_id) | performers → event | Performers assigned to event |

---

## SECTION 10: EVENT STATUS & WORKFLOW

### Status/State Fields

| Field | Possible Values | Usage |
|-------|-----------------|-------|
| approved (tinyint) | 0 (Pending), 1 (Approved) | Event approval workflow |
| order_status (integer) | 0=Pending, 1=Confirmed, 2=Completed, 3=Cancelled | Order fulfillment stage |
| menu_status (integer) | 1=Active, 0=Inactive | Menu item availability |
| employee_status | Active, Inactive, On-Leave, Terminated | Staff availability |
| employee_role | Manager, Staff, Logistics, Chef, Decorator | Job classification |

---

## SECTION 11: INFORMALS & ENTERTAINMENT

### Informal Services (Entertainment Acts, Performances)

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| I_id | VARCHAR(20) | Informal/entertainment service ID | Jeel-Shah |
| I_name | VARCHAR(25) | Service name | Jeel-Shah |
| I_type | VARCHAR(45) | Service type (Music, Comedy, Dance, etc.) | Jeel-Shah |
| I_add | VARCHAR(45) | Service provider address | Jeel-Shah |
| I_mail_id | VARCHAR(20) | Service provider email | Jeel-Shah |
| I_cost | DECIMAL | Service cost | Jeel-Shah |

---

## SECTION 12: AUTHENTICATION & SECURITY

### Account Credentials Tables

| Field Name | Data Type | Description | Source Systems |
|-----------|-----------|-------------|-----------------|
| id | INTEGER | Account identifier | Web-Catering |
| user_name | VARCHAR(45) | Login username | All Three |
| user_password | VARCHAR(45) | Encrypted password | All Three |
| password (customerPass/employeePass) | VARCHAR(30) | Password field in dedicated tables | HxnDev |

**Security Note:** Passwords should be hashed/encrypted in production systems, not stored as plain text.

---

## SECTION 13: UNIQUE FIELD AGGREGATION FOR BEO SYSTEM

### Complete Unified Field Catalog

**Customer Information Fields:**
- Customer ID, First Name, Last Name, Full Name
- Email, Phone Number, Address
- National ID (CNIC), Age
- Account Number, Priority Status
- Username, Password, Discount
- Registration Date

**Event Details Fields:**
- Event ID, Event Name, Event Type
- Event Date, Start Time, End Time
- Number of Guests, Total Cost
- Status (Pending/Approved/Completed)
- Feedback/Notes

**Catering & Menu Fields:**
- Menu ID, Menu Name, Menu Type
- Price, Specialty (Desi/Continental)
- Specific Items: Rice, Bread, Protein, Beverages, Desserts
- Minimum Days, Availability Status
- Food Type Category

**Venue Fields:**
- Venue ID, Name, Location
- Address, City, State, Pincode
- Capacity, Category (Indoor/Outdoor/Riverside)
- Rental Cost, Amenities Description

**Financial Fields:**
- Subtotal, Tax, Total
- Unit Price, Service Charges
- Venue Rent, Transportation Cost
- Photography Charges, Decoration Charges
- Customer Discount

**Personnel Fields:**
- Employee/Staff ID, Name, Contact
- Department, Role, Status
- Wage Type, Wage Rate, Salary
- Performer Type (Dancer, Musician, Filler, Anchor)
- Makeup Artist, Photographer, Decorator Details

**Logistics Fields:**
- Transportation Type, Cost per KM, Seating
- Media Requirements (Photography, Videography, Album, Drone)
- Studio Name, Equipment List
- Supplier Information (Decorators, Caterers, Equipment Vendors)

**Order Management Fields:**
- Order ID, Order Status
- Order Date, Items, Quantities
- Subtotal, Tax, Total
- Customer Reference, Menu Reference

---

## SECTION 14: CSV TEMPLATES FOR DATA IMPORT

### Customer Template
```csv
cust_id,first_name,last_name,email,phone_no,address,cnic,age,account_number,priority_status,username,discount,password,register_date
10001,John,Doe,john@example.com,0321-2565432,123 Main St,12345-678901-2,35,023113566,1,john_doe,0.1,password123,2025-01-01 10:00:00
```

### Event Template
```csv
event_id,name,type,event_date,guests,total_cost,starting_time,ending_time,cust_id,venue_id,studio_id,menu_id,catering_id,media_id,approved
20001,John's Wedding,Wedding,2025-06-15,300,500000,6:00 PM,11:00 PM,10001,88885,44002,10005,75002,56791,1
```

### Menu Template
```csv
menu_id,menu_name,menu_type,price,specialty,days,cost,contact_info
10003,Standard Wedding Package,Desi,60000,Desi,30,60000,0316-7884565
```

### Venue Template
```csv
venue_id,name,location,address,max_capacity,category,contact_info,cost
88885,Aura Grande,Islamabad,Service Road E 11,2000,Indoor,033319922,300000
```

### Staff/Employee Template
```csv
emp_id,name,dob,email,phone_no,cnic,wage_type,wage_rate,points,department,mgr_id
20001,Ahmed Khan,1990-04-12,ahmed@gmail.com,0300-6753456,3323-322938475,Monthly,50000,600,Operations,20000
```

---

## SECTION 15: ENTITY RELATIONSHIP SUMMARY

### Core Entity Relationships

```
CUSTOMER
├── Account (1:1)
├── Orders (1:Many)
│   └── OrderItems (Many:Many via Menu)
└── Addresses (1:Many)

EVENT
├── Customer (Many:1)
├── Venue (Many:1)
├── Catering (Many:1)
├── Menu (Many:1)
├── Studio/Photography (Many:1)
├── Transportation (Many:Many)
├── Employees (Many:Many - appointed_to)
├── Performers (Many:Many - performs_in)
├── Decorators (Many:Many - decorated_by)
├── Equipment (Many:Many - needs)
├── Informals/Entertainment (Many:Many - conducted)
└── Media Requirements (1:1)

MENU
├── Catering (Many:1)
├── OrderItems (1:Many)
└── PriceHistory (1:Many)

EMPLOYEE
├── Manager (ISA relationship)
├── Logistics (ISA relationship)
└── Events (Many:Many - appointed_to)

PERFORMER
├── Dancers (ISA)
├── Musicians (ISA)
├── Fillers (ISA)
├── Anchors (ISA)
└── Events (Many:Many - performs_in)
```

---

## SECTION 16: IMPLEMENTATION RECOMMENDATIONS FOR LARIAT BIBLE

### Recommended Core Tables for The Lariat Catering System

1. **customers** - Customer/account holder information
2. **accounts** - Authentication credentials
3. **events** / **catering_orders** - BEO (Banquet Event Orders)
4. **menu_items** - Available menu options with pricing
5. **order_details** - Line items for each catering order
6. **venues** - Location options
7. **staff** - Internal employee information
8. **vendors** - External service providers (catering, decoration, etc.)
9. **vendor_menu** - Shamrock Foods vs SYSCO pricing comparison
10. **event_assignments** - Staff/vendor assignments to specific events
11. **pricing_history** - Historical cost tracking for margin analysis
12. **equipment_inventory** - Restaurant equipment tracking

### Vendor Comparison Enhancement
```sql
-- Add these fields for Shamrock vs SYSCO tracking:
shamrock_price, shamrock_case_size, shamrock_availability
sysco_price, sysco_case_size, sysco_availability
preferred_vendor, savings_percentage
last_price_update, price_trend
```

### Margin Tracking Table
```sql
CREATE TABLE order_margins (
    order_id INT PRIMARY KEY,
    food_cost DECIMAL(10,2),
    menu_price DECIMAL(10,2),
    margin_percentage DECIMAL(5,2),
    target_margin DECIMAL(5,2),
    margin_variance DECIMAL(5,2),
    catering_flag BOOLEAN
);
```

---

## CONCLUSION

This comprehensive schema aggregates best practices from three GitHub event management systems. Key takeaways:

1. **All three systems** use similar core entities (Customer, Event, Menu, Venue, Staff)
2. **Normalization pattern** splits multi-valued fields (phones, addresses) into separate tables
3. **Status tracking** essential for workflow management (approved, pending, completed)
4. **Cost fields** dispersed but should be consolidated in cost tracking table
5. **Vendor relationships** critical for comparison analysis (add Shamrock vs SYSCO fields)
6. **Many-to-many** associations via junction tables (staff to events, performers to events)

**For The Lariat implementation:**
- Extend with vendor pricing fields
- Add catering-specific margin calculations
- Implement equipment maintenance tracking
- Track ingredient costs from both vendors
- Generate pricing optimization reports
- Monitor monthly savings from vendor selection

---

**Document Version:** 1.0
**Last Updated:** November 18, 2025
**Source Repositories:** 3 GitHub event management systems
**Total Tables Analyzed:** 50+
**Total Unique Fields Extracted:** 200+
