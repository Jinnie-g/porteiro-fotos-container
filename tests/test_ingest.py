from porteiro.ingest import container_code, origin_name, source_id


def test_origin_name_strips_roboflow_and_labelstudio():
    assert origin_name("27ea35ef-DSCF4424_counter_120_jpg") == "DSCF4424_counter_120"
    assert origin_name("DSCF5271_resize_JPG_jpg") == "DSCF5271"
    assert origin_name("aug_50_1350_png_jpg") == "aug_50_1350"
    assert origin_name("NYKU4948678-8-_JPG_jpg") == "NYKU4948678-8-"


def test_container_code_iso6346():
    assert container_code("BMOU5055950-19-") == "BMOU505595"
    assert container_code("921d052f-TCLU2959380_34") == "TCLU295938"
    assert container_code("DSCF5271") is None


def test_source_id():
    assert source_id("aug_51_1678_png_jpg") == "aug_preaumentada"
    assert source_id("CAM12023-02-04-15_01_47-237683_jpg") == "cam_fixa"
    assert source_id("118_jpg") == "numerica"
    assert source_id("CCLU7602510-5-_jpg") == "codigo_iso"
