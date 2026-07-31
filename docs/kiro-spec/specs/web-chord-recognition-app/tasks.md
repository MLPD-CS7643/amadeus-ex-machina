# Implementation Plan: Web Chord Recognition Application

## Overview

This plan breaks down the implementation of a web-based chord recognition system into discrete coding tasks. The system wraps the existing CRNNModel PyTorch implementation with a Flask REST API backend and React frontend. Each task builds incrementally, with early validation through testing and checkpoints to ensure correctness.

## Phase 2 status (React frontend) — COMPLETE

Tasks 11–17 are implemented in `frontend/` (Vite + React 18 + TypeScript + Tailwind
3 + Axios + react-dropzone). 15 Vitest/Testing-Library/fast-check tests pass;
`tsc -b` and `vite build` are clean. Verified live: the Vite dev server proxies
`/api/*` to the Flask backend and the full upload → status → results → download flow
works in-browser.

Deviation: the chord timeline uses a **custom SVG/CSS component**, not Recharts
(Recharts fits poorly for Gantt-style chord segments); it still satisfies design
Properties 12 (positioning) and 13 (distinctness). Remaining: tasks 18 (manual
frontend+backend checkpoint — done informally), 19–23 (Docker + deployment docs).

## Phase 1 status (backend API) — COMPLETE

Tasks 1–10 (the full backend: FileHandler, JobManager, InferenceEngine, Flask
routes, async worker, error handling, rate limiting) are implemented and tested
(`backend/`, 71 passing tests incl. property-based + a real-artifact integration
test). Verified live via curl: upload → status → results → download, plus 400/404/413/429.

Intentional deviations from the original design, forced by the real saved artifacts:
- **Model:** the saved `CRNNModel_best_model.pth` does NOT match `models/CRNN.py`.
  The backend ships a checkpoint-faithful class (`backend/app/model.py`,
  `CRNNChordModel`) recovered from git history (`b12817b`) instead.
- **Features:** the scaler/model need 24-dim Billboard NNLS chroma, which librosa
  can't reproduce. `backend/app/features.py` approximates it with two band-limited
  CQT chromagrams (bass||treble). Accuracy is approximate-but-functional.
- **Inference:** per-frame classification (seq=1), matching how the checkpoint was
  trained on tabular chroma — so `prepare_sequences` was unnecessary.
- **Security:** task 9 rate limiting + filename/error sanitization done; restricted
  upload-dir permissions (9.2) deferred.

Tasks 11–23 (React frontend, Docker, deployment docs) are future phases.

## Tasks

- [x] 1. Set up backend project structure and core dependencies
  - Create backend directory structure (app/, tests/, config/)
  - Create requirements.txt with Flask, Flask-CORS, torch, librosa, scikit-learn dependencies
  - Create .env.example file with configuration variables (MODEL_PATH, SCALER_PATH, ENCODER_PATH, MAX_FILE_SIZE, UPLOAD_DIR)
  - Set up pytest configuration for backend testing
  - _Requirements: 7.1, 7.3_

- [x] 2. Implement FileHandler class for secure file operations
  - [x] 2.1 Create FileHandler class with file validation and sanitization
    - Implement save_upload() method to save uploaded files with sanitized names
    - Implement validate_file() method to check file extension and size
    - Implement sanitize_filename() method to remove path traversal sequences
    - Implement verify_audio_file() method using magic number verification
    - Implement cleanup_file() method for secure file deletion
    - _Requirements: 1.3, 1.4, 1.6, 1.7, 8.4, 10.1, 10.2, 10.5_
  
  - [~] 2.2 Write property test for filename sanitization
    - **Property 23: Filename Sanitization**
    - **Validates: Requirements 10.2**
    - Test that filenames with path traversal sequences are sanitized
  
  - [~] 2.3 Write property test for file validation
    - **Property 1: Invalid File Rejection**
    - **Validates: Requirements 1.4, 1.7, 8.1, 8.2, 8.4, 8.5**
    - Test that invalid files (wrong format, corrupted, empty, oversized) are rejected
  
  - [~] 2.4 Write property test for valid file acceptance
    - **Property 2: Valid File Acceptance**
    - **Validates: Requirements 1.3, 1.5**
    - Test that valid audio files are accepted and metadata is returned
  
  - [~] 2.5 Write property test for file content validation
    - **Property 22: File Content Validation**
    - **Validates: Requirements 10.1**
    - Test that magic number verification detects extension/content mismatches

- [x] 3. Implement JobManager class for job lifecycle tracking
  - [x] 3.1 Create JobManager class with job creation and status tracking
    - Implement create_job() method to generate unique job_id and initialize metadata
    - Implement get_job_status() method to retrieve job information
    - Implement update_job_status() method to update job state and progress
    - Implement cleanup_old_jobs() method to remove expired jobs
    - Implement get_active_job_count() method for concurrency control
    - Create job metadata JSON schema with all required fields
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 6.2, 6.4, 9.5_
  
  - [~] 3.2 Write property test for job status lifecycle
    - **Property 7: Job Status Lifecycle**
    - **Validates: Requirements 3.3, 3.4**
    - Test that job status transitions follow valid state machine
  
  - [~] 3.3 Write property test for job ID uniqueness
    - **Property 14: Job ID Uniqueness**
    - **Validates: Requirements 6.4**
    - Test that multiple uploads generate distinct job IDs
  
  - [~] 3.4 Write unit tests for job cleanup
    - Test that old jobs are removed after timeout threshold
    - Test that temporary files are deleted during cleanup
    - _Requirements: 9.5, 10.5_

- [x] 4. Implement InferenceEngine class for audio processing
  - [x] 4.1 Create InferenceEngine class with model loading
    - Implement load_artifacts() method to load CRNNModel, scaler, and label_encoder
    - Implement device selection logic (GPU/MPS/CPU) using torch.cuda.is_available() and torch.backends.mps.is_available()
    - Add error handling for missing model files
    - _Requirements: 2.3, 2.4, 7.7, 9.2_
  
  - [~] 4.2 Implement chromagram extraction
    - Implement extract_chromagram() method using librosa.feature.chroma_stft
    - Return chromagram features and corresponding timestamps
    - Add error handling for corrupted audio files
    - _Requirements: 2.1, 8.1, 8.2_
  
  - [~] 4.3 Write property test for chromagram extraction consistency
    - **Property 3: Chromagram Extraction Consistency**
    - **Validates: Requirements 2.1**
    - Test that chromagram shape matches audio duration
  
  - [~] 4.4 Implement feature scaling and sequence preparation
    - Implement prepare_sequences() method to create CRNN input sequences
    - Apply scaler transformation to chromagram features
    - _Requirements: 2.2_
  
  - [~] 4.5 Write property test for feature scaling normalization
    - **Property 4: Feature Scaling Normalization**
    - **Validates: Requirements 2.2**
    - Test that scaled features have zero mean and unit variance
  
  - [~] 4.6 Implement model inference and prediction decoding
    - Implement run_inference() method to run CRNNModel forward pass
    - Implement decode_predictions() method to convert class indices to chord labels using label_encoder
    - Map predictions to timestamps to create chord segments
    - _Requirements: 2.5, 2.6_
  
  - [~] 4.7 Write property test for label encoding round trip
    - **Property 5: Label Encoding Round Trip**
    - **Validates: Requirements 2.6**
    - Test that encoding then decoding returns original label
  
  - [~] 4.8 Implement annotation generation and file writing
    - Implement save_annotations() method to write .lab file format
    - Format timestamps with 4 decimal places
    - Ensure annotations cover entire audio duration with no gaps
    - _Requirements: 4.1, 4.2, 4.6_
  
  - [~] 4.9 Write property test for annotation completeness
    - **Property 6: Annotation Generation Completeness**
    - **Validates: Requirements 2.5, 4.1**
    - Test that annotations cover full duration with no gaps or overlaps
  
  - [~] 4.10 Write property test for lab file format compliance
    - **Property 9: Lab File Format Compliance**
    - **Validates: Requirements 4.2, 4.6**
    - Test that .lab file lines match required format with 4 decimal places
  
  - [~] 4.11 Implement process_audio() orchestration method
    - Wire together extract_chromagram, prepare_sequences, run_inference, decode_predictions, save_annotations
    - Add progress callback for status updates
    - Implement error handling for each pipeline stage
    - _Requirements: 2.1, 2.2, 2.5, 2.6, 3.2, 8.3_
  
  - [~] 4.12 Write property test for GPU utilization
    - **Property 19: GPU Utilization When Available**
    - **Validates: Requirements 9.2**
    - Test that model loads on GPU when available

- [x] 5. Checkpoint - Verify inference pipeline with test audio
  - Run InferenceEngine on a sample audio file
  - Verify .lab file is generated with correct format
  - Ensure all tests pass, ask the user if questions arise

- [x] 6. Implement Flask REST API endpoints
  - [x] 6.1 Create Flask application with CORS configuration
    - Initialize Flask app with CORS headers
    - Configure upload directory and file size limits
    - Load environment variables for configuration
    - _Requirements: 7.3, 7.5, 7.6_
  
  - [~] 6.2 Write property test for CORS header presence
    - **Property 17: CORS Header Presence**
    - **Validates: Requirements 7.5**
    - Test that all API responses include CORS headers
  
  - [~] 6.3 Implement POST /api/upload endpoint
    - Accept multipart/form-data file upload
    - Validate file using FileHandler
    - Create job using JobManager
    - Return job_id and status in JSON response
    - Return appropriate HTTP status codes (400, 413, 500)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 6.1, 6.4, 6.5, 6.6_
  
  - [~] 6.4 Write property test for HTTP error code mapping
    - **Property 15: HTTP Error Code Mapping**
    - **Validates: Requirements 6.5, 6.6, 6.7**
    - Test that different error types return correct HTTP status codes
  
  - [~] 6.5 Implement GET /api/status/{job_id} endpoint
    - Retrieve job status from JobManager
    - Return status, progress, estimated time in JSON
    - Return 404 for non-existent jobs
    - _Requirements: 3.1, 3.2, 3.5, 6.2_
  
  - [~] 6.6 Write property test for processing time estimation
    - **Property 8: Processing Time Estimation**
    - **Validates: Requirements 3.5**
    - Test that estimated time is proportional to audio duration
  
  - [~] 6.7 Implement GET /api/results/{job_id} endpoint
    - Retrieve completed job results from JobManager
    - Parse .lab file and convert to JSON format
    - Return annotations array with startTime, endTime, chordLabel
    - Return 404 if job not found or not complete
    - _Requirements: 4.5, 6.3_
  
  - [~] 6.8 Write property test for annotation preview accuracy
    - **Property 11: Annotation Preview Accuracy**
    - **Validates: Requirements 4.5**
    - Test that preview displays correct number of annotations
  
  - [~] 6.9 Implement GET /api/download/{job_id} endpoint
    - Retrieve .lab file for completed job
    - Set Content-Type and Content-Disposition headers
    - Generate filename using pattern {original_name}_chords.lab
    - Return 404 if job not found or not complete
    - _Requirements: 4.3, 4.4, 6.3_
  
  - [~] 6.10 Write property test for output filename pattern
    - **Property 10: Output Filename Pattern**
    - **Validates: Requirements 4.4, 10.2**
    - Test that output filename follows required pattern with sanitization

- [x] 7. Implement asynchronous job processing
  - [x] 7.1 Create background worker for processing jobs
    - Implement job queue mechanism (file-based or in-memory)
    - Create worker thread/process to process jobs asynchronously
    - Implement concurrency limit (max 5 concurrent jobs)
    - Update job status during processing stages
    - _Requirements: 3.2, 3.3, 3.4, 9.3, 9.4_
  
  - [~] 7.2 Write property test for request queuing
    - **Property 20: Request Queuing Beyond Capacity**
    - **Validates: Requirements 9.4**
    - Test that jobs are queued when concurrent limit is reached
  
  - [~] 7.3 Implement error handling and cleanup in worker
    - Catch exceptions during processing and update job status to failed
    - Clean up temporary files on completion or failure
    - Log errors with job_id context
    - _Requirements: 8.3, 9.5, 10.5_
  
  - [~] 7.4 Write property test for temporary file cleanup
    - **Property 21: Temporary File Cleanup**
    - **Validates: Requirements 9.5, 10.5**
    - Test that temporary files are deleted after job completion

- [x] 8. Implement error handling and logging
  - [x] 8.1 Create custom exception classes
    - Define FileValidationError with error codes
    - Define ProcessingError with stage information
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
  
  - [~] 8.2 Implement error response formatting
    - Create error response JSON structure with code, message, details
    - Sanitize error messages to avoid exposing internal paths
    - _Requirements: 8.3, 10.6_
  
  - [~] 8.3 Write property test for error message sanitization
    - **Property 18: Error Message Sanitization**
    - **Validates: Requirements 8.3, 10.6**
    - Test that error messages don't contain sensitive information
  
  - [~] 8.4 Configure logging with appropriate levels
    - Set up logging format with timestamp, level, job_id, component
    - Configure log levels (ERROR, WARNING, INFO, DEBUG)
    - Implement sensitive data redaction in logs
    - _Requirements: 8.3_

- [~] 9. Implement rate limiting and security measures (rate limiting + sanitization done; 9.2 dir perms deferred)
  - [x] 9.1 Add rate limiting middleware
    - Implement rate limiter (10 requests per minute per IP)
    - Return 429 status code when limit exceeded
    - _Requirements: 10.3_
  
  - [~] 9.2 Configure secure file storage
    - Set restricted permissions on upload directory
    - Ensure files are stored outside web root
    - _Requirements: 10.4_
  
  - [~] 9.3 Write unit tests for security measures
    - Test path traversal attack prevention
    - Test magic number spoofing detection
    - Test rate limiting behavior
    - _Requirements: 10.1, 10.2, 10.3_

- [x] 10. Checkpoint - Test backend API with curl/Postman
  - Test file upload endpoint with valid and invalid files
  - Test status polling for a processing job
  - Test results and download endpoints
  - Verify error responses and status codes
  - Ensure all tests pass, ask the user if questions arise

- [x] 11. Set up frontend project structure (Vite instead of CRA; Tailwind 3)
  - Create React app with TypeScript template
  - Install dependencies (axios, recharts, react-dropzone, tailwind)
  - Configure Tailwind CSS
  - Set up project directory structure (components/, types/, api/, utils/)
  - _Requirements: 7.1_

- [x] 12. Implement TypeScript types and API client
  - [x] 12.1 Define TypeScript interfaces
    - Create ChordAnnotation interface
    - Create UploadResponse, StatusResponse, ResultsResponse interfaces
    - Create JobStatus type union
    - _Requirements: 6.1, 6.2, 6.3_
  
  - [~] 12.2 Create API client module
    - Implement uploadFile() function using axios
    - Implement getJobStatus() function with polling
    - Implement getResults() function
    - Implement downloadAnnotations() function
    - Configure axios base URL and error handling
    - _Requirements: 6.1, 6.2, 6.3_

- [x] 13. Implement UploadComponent
  - [x] 13.1 Create file upload UI with drag-and-drop
    - Use react-dropzone for drag-and-drop zone
    - Add file selection button as alternative
    - Display selected filename and file size
    - Show accepted formats and size limit
    - _Requirements: 1.1, 1.2, 1.5_
  
  - [~] 13.2 Implement file validation and upload logic
    - Validate file format (WAV, MP3, FLAC) before upload
    - Validate file size (max 50MB) before upload
    - Display error messages for validation failures
    - Show upload progress indicator
    - Call API client uploadFile() on submit
    - Pass job_id to parent component on success
    - _Requirements: 1.3, 1.4, 1.6, 1.7_
  
  - [~] 13.3 Write unit tests for UploadComponent
    - Test file selection and display
    - Test validation error messages
    - Test upload success flow
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

- [x] 14. Implement StatusComponent
  - [x] 14.1 Create status display UI
    - Display current job status (queued, processing, complete, failed)
    - Show progress indicator or loading animation
    - Display estimated time remaining
    - Show error message if processing fails
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [~] 14.2 Implement status polling logic
    - Poll /api/status endpoint every 2 seconds
    - Update UI with latest status and progress
    - Stop polling when job reaches terminal state
    - Handle network errors with retry logic
    - _Requirements: 3.1, 3.2, 6.2, 8.6_
  
  - [~] 14.3 Write unit tests for StatusComponent
    - Test status display for each state
    - Test polling behavior with mocked API
    - Test error handling
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 15. Implement VisualizationComponent (custom SVG/CSS timeline, not Recharts)
  - [x] 15.1 Create chord timeline visualization
    - Use Recharts to render timeline
    - Position chord segments at correct time coordinates
    - Display chord labels on segments
    - Use distinct colors for different chord types
    - Show total audio duration
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  
  - [~] 15.2 Implement scrolling for long audio files
    - Add horizontal scrollbar for audio > 60 seconds
    - Implement zoom controls (optional)
    - _Requirements: 5.5_
  
  - [~] 15.3 Write property test for visualization data consistency
    - **Property 12: Visualization Data Consistency**
    - **Validates: Requirements 5.1, 5.2, 5.4**
    - Test that timeline positions match annotation timestamps
  
  - [~] 15.4 Write property test for chord visual distinctiveness
    - **Property 13: Chord Visual Distinctiveness**
    - **Validates: Requirements 5.3**
    - Test that different chords have different visual markers
  
  - [~] 15.5 Write unit tests for VisualizationComponent
    - Test rendering with sample annotations
    - Test scrolling behavior
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 16. Implement results display and download
  - [x] 16.1 Create results display component
    - Display preview of first 10 annotations
    - Show download button for .lab file
    - Display filename and audio duration
    - _Requirements: 4.3, 4.4, 4.5_
  
  - [~] 16.2 Implement download functionality
    - Trigger download from /api/download endpoint
    - Set correct filename for downloaded file
    - Handle download errors
    - _Requirements: 4.3, 4.4_
  
  - [~] 16.3 Write unit tests for results component
    - Test annotation preview display
    - Test download button behavior
    - _Requirements: 4.3, 4.4, 4.5_

- [x] 17. Implement main App component and routing
  - [x] 17.1 Create App component with state management
    - Manage job_id state
    - Manage current view state (upload, processing, results)
    - Wire together UploadComponent, StatusComponent, VisualizationComponent
    - Handle state transitions based on job status
    - _Requirements: 1.1, 3.1, 4.1, 5.1_
  
  - [~] 17.2 Implement error boundary
    - Catch and display React errors gracefully
    - Provide "Try Again" button to reset state
    - _Requirements: 8.1, 8.2, 8.3_
  
  - [~] 17.3 Write integration tests for App component
    - Test complete upload-to-results flow with mocked API
    - Test error handling scenarios
    - _Requirements: 1.1, 3.1, 4.1, 5.1_

- [x] 18. Checkpoint - Test frontend with backend running
  - Start backend server
  - Start frontend dev server
  - Test complete user flow: upload → status → visualization → download
  - Test error scenarios (invalid file, network error)
  - Ensure all tests pass, ask the user if questions arise

- [ ] 19. Create Docker configuration
  - [~] 19.1 Create Dockerfile for backend
    - Use Python 3.11 base image
    - Install system dependencies for librosa and PyTorch
    - Copy requirements.txt and install Python dependencies
    - Copy application code
    - Set up model artifacts volume mount
    - Configure Gunicorn as WSGI server
    - Expose port 5000
    - _Requirements: 7.2, 7.6_
  
  - [~] 19.2 Create docker-compose.yml
    - Define backend service with volume mounts for models
    - Configure environment variables
    - Set up network configuration
    - _Requirements: 7.2, 7.3_
  
  - [~] 19.3 Create .dockerignore file
    - Exclude unnecessary files from Docker context
    - _Requirements: 7.2_

- [ ] 20. Implement environment configuration
  - [~] 20.1 Create configuration module
    - Load environment variables with defaults
    - Validate required configuration (model paths exist)
    - _Requirements: 7.3_
  
  - [~] 20.2 Write property test for environment configuration override
    - **Property 16: Environment Configuration Override**
    - **Validates: Requirements 7.3**
    - Test that environment variables override defaults

- [ ] 21. Create deployment documentation
  - [~] 21.1 Write README.md with deployment instructions
    - Document local development setup
    - Document Docker deployment steps
    - Document environment variable configuration
    - Include example .env file
    - Document API endpoints
    - _Requirements: 7.4_
  
  - [~] 21.2 Create API documentation
    - Document all endpoints with request/response examples
    - Document error codes and messages
    - _Requirements: 6.1, 6.2, 6.3_

- [ ] 22. Final integration testing and validation
  - [~] 22.1 Run all property-based tests
    - Execute all property tests with 100 examples each
    - Verify all properties hold
  
  - [~] 22.2 Run all unit tests
    - Execute full test suite for backend and frontend
    - Verify code coverage meets requirements (80% backend, 70% frontend)
  
  - [~] 22.3 Perform end-to-end testing
    - Test with various audio files (different formats, durations)
    - Test concurrent uploads
    - Test error scenarios
    - Verify performance requirements (3-minute audio in <30 seconds)
    - _Requirements: 9.1, 9.3_
  
  - [~] 22.4 Security testing
    - Test path traversal prevention
    - Test magic number verification
    - Test rate limiting
    - Verify error messages don't leak sensitive info
    - _Requirements: 10.1, 10.2, 10.3, 10.6_

- [~] 23. Final checkpoint - Production readiness verification
  - Verify Docker container builds and runs successfully
  - Test deployment with docker-compose
  - Verify all environment variables are documented
  - Confirm all tests pass
  - Review README for completeness
  - Ask the user if questions arise before considering feature complete

## Notes

- Tasks marked with `*` are optional testing tasks and can be skipped for faster MVP delivery
- Each task references specific requirements for traceability
- Property-based tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- Checkpoints ensure incremental validation at key milestones
- The implementation leverages existing CRNNModel and preprocessing artifacts from the codebase
- Backend uses Python/Flask, frontend uses React/TypeScript as specified in design
