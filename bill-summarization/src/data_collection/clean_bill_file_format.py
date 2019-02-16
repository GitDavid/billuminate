import os
import shutil
import zipfile


def clean_bill_file_tree(congress_session, bill_type):
    bills_path = '/Users/melissaferrari/Projects/repo/congress/data/'
    bills_path += '{}/bills/{}'.format(congress_session, bill_type)
    no_xml = []
    no_pdf = []
    bill_folders = os.listdir(bills_path)
    bill_folders = [x for x in bill_folders if not x.startswith('.')]
    for bill_folder in bill_folders:
        print(bill_folder)
        temp_path = os.path.join(bills_path, bill_folder + '/text-versions/')
        folders = [x for x in os.listdir(temp_path) if not x.startswith('.')]
        for folder in folders:
            path_extract = os.path.join(temp_path, folder)
            path_extract_list = [x for x in os.listdir(path_extract)
                                 if not x.startswith('.')]
            if (any(x.startswith('BILLS-') for x in path_extract_list) and
               ('package.zip' in path_extract_list)):
                shutil.rmtree(os.path.join(path_extract,
                                           [x for x in path_extract_list
                                            if x.startswith('BILLS-')][0]))
            new_path = os.path.join(os.path.join(bills_path, bill_folder),
                                    folder)
            os.makedirs(new_path)
            path_to_zip_file = os.path.join(os.path.join(temp_path, folder),
                                            'package.zip')
            zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
            zip_ref.extractall(new_path)
            zip_ref.close()

            path_extracted = os.path.join(new_path, os.listdir(new_path)[0])
            xml_folder_path = os.path.join(path_extracted, 'xml')
            if os.path.exists(xml_folder_path):
                old_xml_path = os.path.join(xml_folder_path,
                                            os.listdir(xml_folder_path)[0])
                os.rename(old_xml_path, os.path.join(new_path,
                          old_xml_path.replace('/xml', '').split('/')[-1]))
                shutil.rmtree(xml_folder_path)
            else:
                no_xml.append(path_extracted)

            # html_folder_path = os.path.join(path_extracted, 'html')

            pdf_folder_path = os.path.join(path_extracted, 'pdf')
            if os.path.exists(pdf_folder_path):
                old_pdf_path = os.path.join(pdf_folder_path,
                                            os.listdir(pdf_folder_path)[0])
                os.rename(old_pdf_path, os.path.join(new_path,
                          old_pdf_path.replace('/pdf', '').split('/')[-1]))
                shutil.rmtree(pdf_folder_path)
            else:
                no_pdf.append(path_extracted)

            os.rename(os.path.join(path_extracted, 'mods.xml'),
                      os.path.join(new_path, 'mods.xml'))
            os.remove(os.path.join(path_extracted, 'premis.xml'))
            os.remove(os.path.join(path_extracted, 'dip.xml'))

            shutil.rmtree(path_extracted)
        shutil.rmtree(temp_path)
    return no_xml, no_pdf


if __name__ == "__main__":
    congress_session = 112
    print(congress_session)
    bill_types = ['hr', 's']
    for bill_type in bill_types[1:]:
        no_xml, no_pdf = clean_bill_file_tree(congress_session, bill_type)
        print(congress_session, bill_type)
        print('no_xml')
        print(no_xml)
        print('no_pdf')
        print(no_pdf)
