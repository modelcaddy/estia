//! Long-running work with an id a client can poll or watch: model pulls and
//! the runtime install.

use estia_engine::models::DownloadProgress;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use tokio::sync::watch;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Running,
    Done,
    Failed,
}

#[derive(Debug, Clone, Serialize)]
pub struct JobView {
    pub id: String,
    pub kind: &'static str,
    pub model_id: String,
    pub status: JobStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<DownloadProgress>,
    /// Runtime install phase (`kind == "runtime"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub setup: Option<estia_engine::runtime::SetupProgress>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
}

pub struct Job {
    pub view: watch::Sender<JobView>,
}

#[derive(Default)]
pub struct JobTable {
    next: AtomicU64,
    jobs: Mutex<HashMap<String, Job>>,
}

impl JobTable {
    pub fn create(&self, kind: &'static str, model_id: &str) -> (String, watch::Sender<JobView>) {
        let n = self.next.fetch_add(1, Ordering::SeqCst) + 1;
        let id = format!("job_{n}_{}", std::process::id());
        let view = JobView {
            id: id.clone(),
            kind,
            model_id: model_id.to_string(),
            status: JobStatus::Running,
            progress: None,
            setup: None,
            error: None,
            result: None,
        };
        let (tx, _rx) = watch::channel(view);
        self.jobs.lock().unwrap().insert(id.clone(), Job { view: tx.clone() });
        (id, tx)
    }

    pub fn get(&self, id: &str) -> Option<JobView> {
        self.jobs.lock().unwrap().get(id).map(|j| j.view.borrow().clone())
    }

    pub fn subscribe(&self, id: &str) -> Option<watch::Receiver<JobView>> {
        self.jobs.lock().unwrap().get(id).map(|j| j.view.subscribe())
    }

    /// The running pull for `model_id`, if one exists.
    pub fn running_pull(&self, model_id: &str) -> Option<JobView> {
        self.running("pull", model_id)
    }

    /// The running job of `kind` for `model_id`, if one exists.
    pub fn running(&self, kind: &str, model_id: &str) -> Option<JobView> {
        self.jobs
            .lock()
            .unwrap()
            .values()
            .map(|j| j.view.borrow().clone())
            .find(|v| v.kind == kind && v.model_id == model_id && matches!(v.status, JobStatus::Running))
    }

    pub fn list(&self) -> Vec<JobView> {
        self.jobs.lock().unwrap().values().map(|j| j.view.borrow().clone()).collect()
    }
}
