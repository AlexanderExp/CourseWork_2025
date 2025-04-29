from pkg_resources import parse_requirements
from setuptools import find_packages, setup


def main():
    package_name = 'mlpt'
    # Ищем пакеты в каталоге mlpt
    # Если не находится ultralytics, нужно ввести в терминале pip install -e mlpt/modules/ultralytics
    packages = find_packages(where="mlpt", include=["mlpt.modules.ultralytics*"])
    print("Найденные пакеты:", packages)


    
    # Чтение зависимостей из requirements.txt с указанием кодировки UTF-8
    with open('requirements.txt', encoding='utf-8') as f:
        reqs = [str(req) for req in parse_requirements(f)]
    
    setup(
        name=package_name,
        version='0.0.1',
        author='HASKII',
        description='Detection',
        package_dir={"": package_name},
        packages=packages,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.7',
        install_requires=reqs,
    )


if __name__ == '__main__':
    main()
