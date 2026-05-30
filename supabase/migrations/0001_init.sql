-- sys-dyna v2.0 initial schema (Supabase / Postgres)
-- Replaces the v1.0 SQLite schema. Auth is delegated to Supabase Auth
-- (auth.users, Google SSO); application data lives here with RLS so each
-- user only sees their own rows. An "admin" role may read across the org.
--
-- Design ref: docs/design_v2.md sections 7 (data model) and 8 (auth).

-- ---------------------------------------------------------------------------
-- profiles: app-side user attributes, 1:1 with auth.users
-- Created first so the is_admin() SQL function below (whose body is validated
-- at creation time) can reference it on a fresh database.
-- ---------------------------------------------------------------------------
create table if not exists public.profiles (
    user_id      uuid primary key references auth.users(id) on delete cascade,
    display_name text not null,
    department   text,
    role         text not null default 'member' check (role in ('member', 'admin')),
    created_at   timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- Helper: is the current user an admin? (admins get org-wide read access)
-- ---------------------------------------------------------------------------
create or replace function public.is_admin()
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select coalesce(
    (select p.role = 'admin' from public.profiles p where p.user_id = auth.uid()),
    false
  );
$$;

alter table public.profiles enable row level security;

create policy profiles_select_self_or_admin on public.profiles
    for select using (user_id = auth.uid() or public.is_admin());
-- Self-service rows are forced to role 'member'; only admins may grant 'admin'.
-- This blocks privilege escalation via a self-update of the role column.
create policy profiles_insert_self_member on public.profiles
    for insert with check (user_id = auth.uid() and role = 'member');
create policy profiles_update_self_no_escalation on public.profiles
    for update using (user_id = auth.uid())
    with check (user_id = auth.uid() and role <> 'admin');
create policy profiles_update_admin on public.profiles
    for update using (public.is_admin()) with check (true);

-- ---------------------------------------------------------------------------
-- sd_models: catalog + user-uploaded model registry
--   built-in catalog models are also code-defined (src/.../simulation/catalog.py);
--   this table tracks uploads and any DB-managed catalog entries.
-- ---------------------------------------------------------------------------
create table if not exists public.sd_models (
    model_id     text primary key,
    owner_id     uuid references auth.users(id) on delete set null,
    name         text not null,
    description  text,
    source       text not null check (source in ('catalog', 'upload')),
    storage_path text,                       -- Supabase Storage path for uploads
    params       jsonb not null default '[]'::jsonb,
    created_at   timestamptz not null default now()
);

alter table public.sd_models enable row level security;

-- Catalog models are visible to everyone; uploads only to their owner/admin.
create policy sd_models_select on public.sd_models
    for select using (
        source = 'catalog' or owner_id = auth.uid() or public.is_admin()
    );
-- Uploads belong to their owner; only admins may publish shared catalog models.
create policy sd_models_insert on public.sd_models
    for insert with check (
        (source = 'upload' and owner_id = auth.uid())
        or (source = 'catalog' and public.is_admin())
    );
create policy sd_models_update on public.sd_models
    for update using (owner_id = auth.uid() or public.is_admin())
    with check (
        (source = 'upload' and owner_id = auth.uid())
        or public.is_admin()
    );
create policy sd_models_delete_own on public.sd_models
    for delete using (owner_id = auth.uid() or public.is_admin());

-- ---------------------------------------------------------------------------
-- sessions: one conversation
-- ---------------------------------------------------------------------------
create table if not exists public.sessions (
    session_id  uuid primary key default gen_random_uuid(),
    user_id     uuid not null references auth.users(id) on delete cascade,
    title       text,
    model_name  text,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);
create index if not exists ix_sessions_user on public.sessions(user_id);

alter table public.sessions enable row level security;

create policy sessions_rw_own on public.sessions
    for all using (user_id = auth.uid() or public.is_admin())
    with check (user_id = auth.uid());

-- ---------------------------------------------------------------------------
-- messages: chat log (normalised from v1.0 chat_log JSON)
-- ---------------------------------------------------------------------------
create table if not exists public.messages (
    message_id bigint generated always as identity primary key,
    session_id uuid not null references public.sessions(session_id) on delete cascade,
    role       text not null check (role in ('user', 'assistant', 'system', 'tool')),
    content    text not null default '',
    created_at timestamptz not null default now()
);
create index if not exists ix_messages_session on public.messages(session_id);

alter table public.messages enable row level security;

-- Visibility is inherited from the owning session.
create policy messages_rw_via_session on public.messages
    for all using (
        exists (
            select 1 from public.sessions s
            where s.session_id = messages.session_id
              and (s.user_id = auth.uid() or public.is_admin())
        )
    )
    with check (
        exists (
            select 1 from public.sessions s
            where s.session_id = messages.session_id
              and s.user_id = auth.uid()
        )
    );

-- ---------------------------------------------------------------------------
-- simulation_runs: one execution (model + scenarios)
-- ---------------------------------------------------------------------------
create table if not exists public.simulation_runs (
    run_id      uuid primary key default gen_random_uuid(),
    session_id  uuid not null references public.sessions(session_id) on delete cascade,
    user_id     uuid not null references auth.users(id) on delete cascade,
    model_id    text not null,
    scenarios   jsonb not null default '[]'::jsonb,   -- [{name, params}]
    created_at  timestamptz not null default now()
);
create index if not exists ix_runs_session on public.simulation_runs(session_id);

alter table public.simulation_runs enable row level security;

create policy runs_rw_own on public.simulation_runs
    for all using (user_id = auth.uid() or public.is_admin())
    with check (user_id = auth.uid());

-- ---------------------------------------------------------------------------
-- simulation_results: time-series output per run (JSONB)
-- ---------------------------------------------------------------------------
create table if not exists public.simulation_results (
    result_id        uuid primary key default gen_random_uuid(),
    run_id           uuid not null references public.simulation_runs(run_id) on delete cascade,
    time_series_data jsonb not null,           -- {scenario: {var: [{t,v}]}}
    created_at       timestamptz not null default now()
);
create index if not exists ix_results_run on public.simulation_results(run_id);

alter table public.simulation_results enable row level security;

create policy results_rw_via_run on public.simulation_results
    for all using (
        exists (
            select 1 from public.simulation_runs r
            where r.run_id = simulation_results.run_id
              and (r.user_id = auth.uid() or public.is_admin())
        )
    )
    with check (
        exists (
            select 1 from public.simulation_runs r
            where r.run_id = simulation_results.run_id
              and r.user_id = auth.uid()
        )
    );

-- ---------------------------------------------------------------------------
-- tool_call_logs: telemetry. RLS-restricted to the owning user (design v2 §8);
-- the v1.0 "visible to all employees" assumption is dropped.
-- ---------------------------------------------------------------------------
create table if not exists public.tool_call_logs (
    log_id      uuid primary key default gen_random_uuid(),
    session_id  uuid not null references public.sessions(session_id) on delete cascade,
    user_id     uuid not null references auth.users(id) on delete cascade,
    tool_name   text not null,
    tool_input  jsonb,
    tool_output jsonb,
    called_at   timestamptz not null default now(),
    duration_ms integer
);
create index if not exists ix_tool_logs_session on public.tool_call_logs(session_id);
create index if not exists ix_tool_logs_tool on public.tool_call_logs(tool_name);

alter table public.tool_call_logs enable row level security;

create policy tool_logs_select_own on public.tool_call_logs
    for select using (user_id = auth.uid() or public.is_admin());
create policy tool_logs_insert_own on public.tool_call_logs
    for insert with check (user_id = auth.uid());
