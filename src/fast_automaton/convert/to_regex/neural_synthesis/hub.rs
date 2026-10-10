//! Downloads models from the Hugging Face Hub.
//!
//! Files go through the Hub's local cache (`HF_HUB_CACHE`, else
//! `HF_HOME/hub`, else `~/.cache/huggingface/hub`), so a model is downloaded
//! once. `HF_TOKEN` authenticates for private models, and `HF_ENDPOINT`
//! points to another Hub.

use std::path::PathBuf;

use super::{Device, NeuralSynthesisError, NeuralSynthesizer};

impl NeuralSynthesizer {
    /// The model [`from_hub`](Self::from_hub) loads by default.
    pub const DEFAULT_HUB_MODEL: &str = "alexvbrdn/kleene1-9m-b16-t128";

    /// Downloads a model from the Hugging Face Hub (its latest revision) and
    /// loads it. `repo_id` is `owner/name`, e.g.
    /// [`DEFAULT_HUB_MODEL`](Self::DEFAULT_HUB_MODEL).
    pub fn from_hub(repo_id: &str, device: Device) -> Result<Self, NeuralSynthesisError> {
        Self::from_hub_revision(repo_id, "main", device)
    }

    /// Downloads a model from the Hugging Face Hub at `revision` (a branch,
    /// a release tag such as `v1.0`, or a commit hash) and loads it.
    pub fn from_hub_revision(
        repo_id: &str,
        revision: &str,
        device: Device,
    ) -> Result<Self, NeuralSynthesisError> {
        let [config, weights] = download(repo_id, revision, ["config.json", "model.safetensors"])?;
        Self::from_files(config, weights, device)
    }
}

fn download<const N: usize>(
    repo_id: &str,
    revision: &str,
    files: [&str; N],
) -> Result<[PathBuf; N], NeuralSynthesisError> {
    let hub_error = |err: hf_hub::HFError| NeuralSynthesisError::Hub(format!("{repo_id}: {err}"));
    let Some((owner, name)) = repo_id.split_once('/') else {
        return Err(NeuralSynthesisError::Hub(format!(
            "{repo_id}: expected a repository id of the form owner/name"
        )));
    };
    let repository = hf_hub::HFClientSync::new()
        .map_err(hub_error)?
        .model(owner, name);
    let mut paths = files.map(|_| PathBuf::new());
    for (path, file) in paths.iter_mut().zip(files) {
        *path = repository
            .download_file()
            .filename(file)
            .revision(revision.to_string())
            .send()
            .map_err(hub_error)?;
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_malformed_repository_ids() {
        assert!(matches!(
            NeuralSynthesizer::from_hub("kleene1-9m-b16-t128", Device::Cpu),
            Err(NeuralSynthesisError::Hub(_))
        ));
    }
}
