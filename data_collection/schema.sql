-- CREATE DATABASE congressional_bills;

CREATE TABLE bills (
    id  SERIAL PRIMARY KEY,
    --- name VARCHAR,

    enacted_as VARCHAR(250),

    awaiting_signature BOOLEAN,
    enacted BOOLEAN,
    active BOOLEAN,
    vetoed BOOLEAN,

    official_title VARCHAR(2000),
    popular_title VARCHAR(2000),
    url VARCHAR(1000),
    
    bill_type INT,
    status_at DATE,
    by_request BOOLEAN,

    sponsor INT,
    sponsor_type VARCHAR(500), -- check for types

    updated_at TIMESTAMP,
    status INT,

    number INT,
    subjects_top_term VARCHAR(500),
    bill_id VARCHAR(12),
    introduced_at TIMESTAMP,

    congress INT,
    
    short_title VARCHAR(250)

    -- ammendments
    -- commitee_reports
);

ALTER TABLE bills
ADD COLUMN enacted_as_type VARCHAR(7);

ALTER TABLE bills
ADD COLUMN enacted_as_number INT;



CREATE TABLE lk_status (  -- lookup table
    id SERIAL PRIMARY KEY,
    name VARCHAR(50)
);

INSERT INTO lk_status (name)
VALUES 
    ('ENACTED:SIGNED'), 
    ('PASS_OVER:HOUSE'), 
    ('REPORTED'), 
    ('REFERRED'),
    ('PROV_KILL:CLOTUREFAILED'),
    ('PROV_KILL:SUSPENSIONFAILED'),
    ('PASS_BACK:SENATE'), 
    ('PROV_KILL:VETO'), 
    ('CONFERENCE:PASSED:HOUSE'),
    ('FAIL:ORIGINATING:HOUSE'), 
    ('VETOED:OVERRIDE_FAIL_ORIGINATING:HOUSE'),
    ('ENACTED:TENDAYRULE');

CREATE TABLE lk_bill_type (  -- lookup table
    id SERIAL PRIMARY KEY,
    name VARCHAR(8)
);

INSERT INTO lk_bill_type (name)
VALUES
    ('hr'), 
    ('hres'), 
    ('hjres'), 
    ('hconres'), 
    ('s'), 
    ('sres'), 
    ('sjres'), 
    ('sconres');

CREATE TABLE related_bills (
    bill_id INT,
    related_bill_id INT,
    identified_by VARCHAR(250), -- check for types
    reason VARCHAR(250),  -- check for types
    "type" VARCHAR(250) -- check for types
);

CREATE TABLE congresspeople (
    id SERIAL PRIMARY KEY,
    bioguide_id VARCHAR(10),
    name VARCHAR(50),
    state VARCHAR(2),
    title VARCHAR(5),
    district INT,
    party CHAR
);

CREATE TABLE bill_actions (
    id SERIAL PRIMARY KEY,
    action_code VARCHAR(50), -- check for types
    -- references 
    text VARCHAR(500),
    type VARCHAR(50) -- check for types
);

CREATE TABLE committees (
    id SERIAL PRIMARY KEY,
    committee_name VARCHAR(250),
    committee_id VARCHAR(5) --check for type
);

CREATE TABLE bill_committee (
    bill_id INT,
    committee_primary_id INT
    -- activity list?
);

CREATE TABLE action_committee (
    action_primary_id INT,
    committee_primary_id INT
);

CREATE TABLE subjects (
    bill_id INT,
    subject VARCHAR(250)
);

CREATE TABLE cosponsorship (
    bill_id INT,
    congresspeople_id INT,
    original_cosponsor BOOLEAN,
    sponsored_at DATE,
    withdrawn_at DATE
);

CREATE TABLE summaries (
    id SERIAL PRIMARY KEY,
    bill_id INT NOT NULL,
    "as" VARCHAR(100),
    date TIMESTAMP,
    text TEXT,
    source INT
);

CREATE TABLE lk_summary_source (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50)
);

INSERT INTO lk_summary_source (name)
VALUES 
    ('CRS'), 
    ('GOVTRACK');

CREATE TABLE bill_text (
    bill_id INT,
    text TEXT
);