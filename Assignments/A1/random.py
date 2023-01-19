# write your code here
import os
import re

def get_earliest_grad_year(tag_text):
  grad_year = ''
  # Define a regular expression pattern to search for years
  pattern = r'\b(19[0-9][0-9]|20[0-9][0-9])\b'
  # Use the finditer method to find all matches
  grad_years = [int(match.group(1)) for match in re.finditer(pattern, tag_text)]
  if grad_years:
    grad_year = min(grad_years)
  return grad_year

names = []
earliest_grad_years = []
for profile in os.listdir(faculty_profiles_dir):
    if profile.endswith('.txt'):
        profile_path = os.path.join(faculty_profiles_dir, profile)
        name = ''
        earliest_grad_year = ''
        with open(faculty_profiles_dir + profile) as f:
           csfaculty_profile = f.read()
        html_soup = BeautifulSoup(csfaculty_profile, 'html.parser')

        faculty_name_container = html_soup.find('div', class_ = 'title section')
        if faculty_name_container.h1.text != None:
          name = faculty_name_container.h1.text
        names.append(name.title())

        faculty_education_containers = html_soup.find_all('div', class_ = 'text parbase section')
        for faculty_education_container in faculty_education_containers:
          if (faculty_education_container != None and 'Education' in faculty_education_container.text):
            earliest_grad_year = get_earliest_grad_year(faculty_education_container.text)
        earliest_grad_years.append(earliest_grad_year)

faculty_grad_year = pd.DataFrame.from_dict({
    'name': names,
    'gradyear': earliest_grad_years
    })
faculty_grad_year.to_csv('faculty_grad_year.csv', index=False)
faculty_grad_year