---
layout: single
title: "Commit message guide"
date: 2025-08-18
categories: [Data Structure]
toc: true      # 목차 표시 여부
---

# Commit message guide
- 깃허브 페이지를 사용하면서 `commit`을 사용해야 되는 경우가 많아짐
- 일반적으로 사용되고 더 실용적인 형태의 commit message를 사용하는 것이 필요해보여서 포스팅함

## Form
```
<타입>: <변경 요약(제목)>

<본문 - 선택, 필요할 때>
```
- 일반적인 commit message의 형태로 예를 들어 아래와 같이 작성함

    ```
    feat: add user login API

    Implemented new login endpoint using JWT auth. Handles  validation,
    returns appropriate error messages, and updates user session.

    Related to #21
    ```
- cli 작성법
    - `git commit -m "feat: add user login API"`
    - `-m` 옵션을 여러 번 써서 제목과 본문을 구분할 수도 있음
        ```
        git commit -m "feat: add commit guide post"\
                -m "Refer https://www.freecodecamp.org/news/writing-good-commit-messages-a-practical-guide/"
        ```

## Commit type
- `feat`: 새로운 기능 추가
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 코드 포맷팅(비기능적), 시각적 요소와 연관된 feature
- `refactor`: 코드 리팩터링
- `test`: 테스트 추가/수정
- `chore`: 기타 변경(빌드, 패키지 등)

## Rules
1. 커밋의 종류 명시
2. 제목을 본문으로부터 한 줄 띄워 구분
3. 불필요하나 구두점 삭제
4. 제목 행을 마침표로 끝내지 말것
5. 제목 행에는 명령어 사용
6. 최대한 자세하고 타인의 시선에서 이해할 수 있도록 작성