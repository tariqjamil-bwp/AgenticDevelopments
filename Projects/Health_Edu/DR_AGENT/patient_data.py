import os
import streamlit as st

# Target folder
PATIENT_DATA_FOLDER = "patient_data"
os.makedirs(PATIENT_DATA_FOLDER, exist_ok=True)


def save_uploaded_files(files, prefix):
    """Save uploaded files into patient_data with tagging prefix."""
    saved_files = []
    for file in files:
        filename = f"{prefix}_{file.name}"
        filepath = os.path.join(PATIENT_DATA_FOLDER, filename)
        with open(filepath, "wb") as f:
            f.write(file.getbuffer())
        saved_files.append(filename)
    return saved_files


def main():
    st.title("🩺 Patient Data Intake Form")

    st.write("Please upload the patient's data in the following categories. "
             "Files will be tagged and saved in `patient_data/` folder.")

    # 1. Previous Prescription
    st.header("1. Previous Prescription")
    prescription_files = st.file_uploader("Upload prescription files (PDF/Image)", 
                                          type=["pdf", "png", "jpg", "jpeg"], 
                                          accept_multiple_files=True,
                                          key="prescription")
    if prescription_files:
        saved = save_uploaded_files(prescription_files, "PRESCRIPTION")
        st.success(f"Saved {len(saved)} prescription file(s): {', '.join(saved)}")

    # 2. Medicines Snapshot
    st.header("2. Medicines Snapshot")
    medicine_files = st.file_uploader("Upload medicine snapshot(s)", 
                                      type=["png", "jpg", "jpeg"], 
                                      accept_multiple_files=True,
                                      key="medicine")
    if medicine_files:
        saved = save_uploaded_files(medicine_files, "MEDICINE")
        st.success(f"Saved {len(saved)} medicine snapshot(s): {', '.join(saved)}")

    # 3. Lab Reports
    st.header("3. Lab Reports")
    lab_files = st.file_uploader("Upload lab reports (PDF/Image)", 
                                 type=["pdf", "png", "jpg", "jpeg"], 
                                 accept_multiple_files=True,
                                 key="lab")
    if lab_files:
        saved = save_uploaded_files(lab_files, "LAB")
        st.success(f"Saved {len(saved)} lab report(s): {', '.join(saved)}")

    # 4. Imaging Data
    st.header("4. Imaging Data (X-Ray / Ultrasound / MRI / CT Scan)")
    imaging_files = st.file_uploader("Upload imaging files (PDF/Image)", 
                                     type=["pdf", "png", "jpg", "jpeg"], 
                                     accept_multiple_files=True,
                                     key="imaging")
    if imaging_files:
        saved = save_uploaded_files(imaging_files, "IMAGING")
        st.success(f"Saved {len(saved)} imaging file(s): {', '.join(saved)}")

    st.write("---")
    st.info("✅ All uploaded files are saved in `patient_data/` and tagged with their category.")


if __name__ == "__main__":
    main()
