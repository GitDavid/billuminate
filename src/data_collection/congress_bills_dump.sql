--
-- PostgreSQL database dump
--

-- Dumped from database version 9.6.11
-- Dumped by pg_dump version 9.6.11

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: congressional_bills; Type: DATABASE; Schema: -; Owner: postgres
--

CREATE DATABASE congressional_bills WITH TEMPLATE = template0 ENCODING = 'UTF8' LC_COLLATE = 'C.UTF-8' LC_CTYPE = 'C.UTF-8';


ALTER DATABASE congressional_bills OWNER TO postgres;

\connect congressional_bills

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: action_committee; Type: TABLE; Schema: public; Owner: melissaferrari
--

CREATE TABLE public.action_committee (
    action_primary_id integer,
    committee_primary_id integer
);


ALTER TABLE public.action_committee OWNER TO melissaferrari;

--
-- Name: bill_actions; Type: TABLE; Schema: public; Owner: melissaferrari
--

CREATE TABLE public.bill_actions (
    id integer NOT NULL,
    action_code character varying(50),
    text character varying(500),
    type character varying(50)
);


ALTER TABLE public.bill_actions OWNER TO melissaferrari;

--
-- Name: bill_committee; Type: TABLE; Schema: public; Owner: melissaferrari
--

CREATE TABLE public.bill_committee (
    bill_id integer,
    committee_primary_id integer
);


ALTER TABLE public.bill_committee OWNER TO melissaferrari;

--
-- Name: bill_text; Type: TABLE; Schema: public; Owner: melissaferrari
--

CREATE TABLE public.bill_text (
    bill_ix integer,
    bill_version_id integer,
    text text
);


ALTER TABLE public.bill_text OWNER TO melissaferrari;

--
-- Name: bill_versions; Type: TABLE; Schema: public; Owner: melissaferrari
--

CREATE TABLE public.bill_versions (
    id bigint,
    title text,
    code text,
    definition text,
    chamber text
);


ALTER TABLE public.bill_versions OWNER TO melissaferrari;

--
-- Name: bills; Type: TABLE; Schema: public; Owner: melissaferrari
--

CREATE TABLE public.bills (
    id integer NOT NULL,
    official_title character varying(2000),
    popular_title character varying(2000),
    url character varying(1000),
    bill_type integer,
    status_at date,
    by_request boolean,
    sponsor integer,
    updated_at timestamp without time zone,
    status integer,
    number integer,
    subjects_top_term character varying(500),
    bill_id character varying(12),
    introduced_at timestamp without time zone,
    congress integer,
    short_title character varying(1000)
);


ALTER TABLE public.bills OWNER TO melissaferrari;

--
-- Name: committees; Type: TABLE; Schema: public; Owner: melissaferrari
--

CREATE TABLE public.committees (
    id integer NOT NULL,
    committee_name character varying(250),
    committee_id character varying(5)
);


ALTER TABLE public.committees OWNER TO melissaferrari;

--
-- Name: cosponsorship; Type: TABLE; Schema: public; Owner: melissaferrari
--

CREATE TABLE public.cosponsorship (
    bill_ix integer,
    legislator_ix integer,
    original_cosponsor boolean
);


ALTER TABLE public.cosponsorship OWNER TO melissaferrari;

--
-- Name: legislators; Type: TABLE; Schema: public; Owner: melissaferrari
--

CREATE TABLE public.legislators (
    id bigint,
    last_name text,
    first_name text,
    middle_name text,
    suffix text,
    nickname text,
    full_name text,
    birthday text,
    gender text,
    type text,
    state text,
    district double precision,
    senate_class double precision,
    party text,
    url text,
    address text,
    phone text,
    contact_form text,
    rss_url text,
    twitter text,
    facebook text,
    youtube text,
    youtube_id text,
    bioguide_id text,
    thomas_id double precision,
    opensecrets_id text,
    lis_id text,
    fec_ids text,
    cspan_id double precision,
    govtrack_id bigint,
    votesmart_id double precision,
    ballotpedia_id text,
    washington_post_id double precision,
    icpsr_id double precision,
    wikipedia_id text,
    current_legislator boolean
);


ALTER TABLE public.legislators OWNER TO melissaferrari;

--
-- Name: lk_bill_type; Type: TABLE; Schema: public; Owner: melissaferrari
--

CREATE TABLE public.lk_bill_type (
    id bigint,
    name character varying(8)
);


ALTER TABLE public.lk_bill_type OWNER TO melissaferrari;

--
-- Name: lk_status; Type: TABLE; Schema: public; Owner: melissaferrari
--

CREATE TABLE public.lk_status (
    id integer NOT NULL,
    name character varying(50)
);


ALTER TABLE public.lk_status OWNER TO melissaferrari;

--
-- Name: lk_summary_source; Type: TABLE; Schema: public; Owner: melissaferrari
--

CREATE TABLE public.lk_summary_source (
    id integer NOT NULL,
    name character varying(50)
);


ALTER TABLE public.lk_summary_source OWNER TO melissaferrari;

--
-- Name: related_bills; Type: TABLE; Schema: public; Owner: melissaferrari
--

CREATE TABLE public.related_bills (
    bill_ix integer,
    related_bill_ix integer,
    identified_by character varying(250),
    reason character varying(250)
);


ALTER TABLE public.related_bills OWNER TO melissaferrari;

--
-- Name: subjects; Type: TABLE; Schema: public; Owner: melissaferrari
--

CREATE TABLE public.subjects (
    bill_ix integer,
    subject character varying(250)
);


ALTER TABLE public.subjects OWNER TO melissaferrari;

--
-- Name: summaries; Type: TABLE; Schema: public; Owner: melissaferrari
--

CREATE TABLE public.summaries (
    id integer NOT NULL,
    bill_ix integer NOT NULL,
    "as" character varying(100),
    date timestamp without time zone,
    text text,
    source integer
);


ALTER TABLE public.summaries OWNER TO melissaferrari;

--
-- Name: bill_actions bill_actions_pkey; Type: CONSTRAINT; Schema: public; Owner: melissaferrari
--

ALTER TABLE ONLY public.bill_actions
    ADD CONSTRAINT bill_actions_pkey PRIMARY KEY (id);


--
-- Name: bills bills_pkey; Type: CONSTRAINT; Schema: public; Owner: melissaferrari
--

ALTER TABLE ONLY public.bills
    ADD CONSTRAINT bills_pkey PRIMARY KEY (id);


--
-- Name: committees committees_pkey; Type: CONSTRAINT; Schema: public; Owner: melissaferrari
--

ALTER TABLE ONLY public.committees
    ADD CONSTRAINT committees_pkey PRIMARY KEY (id);


--
-- Name: lk_status lk_status_pkey; Type: CONSTRAINT; Schema: public; Owner: melissaferrari
--

ALTER TABLE ONLY public.lk_status
    ADD CONSTRAINT lk_status_pkey PRIMARY KEY (id);


--
-- Name: lk_summary_source lk_summary_source_pkey; Type: CONSTRAINT; Schema: public; Owner: melissaferrari
--

ALTER TABLE ONLY public.lk_summary_source
    ADD CONSTRAINT lk_summary_source_pkey PRIMARY KEY (id);


--
-- Name: summaries summaries_pkey; Type: CONSTRAINT; Schema: public; Owner: melissaferrari
--

ALTER TABLE ONLY public.summaries
    ADD CONSTRAINT summaries_pkey PRIMARY KEY (id);


--
-- Name: ix_bill_versions_id; Type: INDEX; Schema: public; Owner: melissaferrari
--

CREATE INDEX ix_bill_versions_id ON public.bill_versions USING btree (id);


--
-- Name: ix_legislators_id; Type: INDEX; Schema: public; Owner: melissaferrari
--

CREATE INDEX ix_legislators_id ON public.legislators USING btree (id);


--
-- Name: ix_lk_bill_type_id; Type: INDEX; Schema: public; Owner: melissaferrari
--

CREATE INDEX ix_lk_bill_type_id ON public.lk_bill_type USING btree (id);


--
-- PostgreSQL database dump complete
--

